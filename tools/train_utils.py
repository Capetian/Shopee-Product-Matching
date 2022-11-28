from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold

# PATH = Path('../data')
PATH = Path('data')

def no_split(df):
    dummy_df = df.copy()
    dummy_df['is_valid'] = False
    df['is_valid'] = True
    df= pd.concat([df, dummy_df])
    return df


def add_splits(train_df, valid_group=0):
    grouped = train_df.groupby('label_group').size()

    labels, sizes =grouped.index.to_list(), grouped.to_list()

    skf = StratifiedKFold(5)
    splits = list(skf.split(labels, sizes))

    group_to_split =  dict()
    for idx in range(5):
        labs = np.array(labels)[splits[idx][1]]
        group_to_split.update(dict(zip(labs, [idx]*len(labs))))

    train_df['split'] = train_df.label_group.replace(group_to_split)
    train_df['is_valid'] = train_df['split'] == valid_group
    return train_df


        
def get_img_file( row):
    img =row.image
    fn  = PATH/'train_images'/img
    if not fn.is_file():
        fn = PATH/'test_images'/img
    return fn


def get_dls(df, size=224, bs=64):
    data_block = DataBlock(blocks = (ImageBlock(), CategoryBlock(vocab=df.label_group.to_list())),
                 splitter=ColSplitter(),
                 get_y=ColReader('label_group'),
                 get_x=get_img_file,
                 item_tfms=Resize(int(size*2), resamples=(Image.BICUBIC,Image.BICUBIC)),
                 
                 batch_tfms=aug_transforms(size=size, min_scale=0.75)+[Normalize.from_stats(*imagenet_stats)]
                 + [Dihedral(p=0.5,draw=1)],
                 )
    return data_block.dataloaders(df, bs=bs)

def get_image_dls(df, path, valid_col=None, size=224, bs=64,shuffle=False, train=True):
    batch_tfms = aug_transforms(size=size, min_scale=0.75) + [Normalize.from_stats(*imagenet_stats)] 
    if train : batch_tfms = batch_tfms  + [Dihedral(p=0.5,draw=1)]
    dls = ImageDataLoaders.from_df(
                        df, path=path, seed=1,
                        fn_col=1, label_col=4, valid_col=valid_col,
                        item_tfms=Resize(int(size*2), resamples=(Image.BICUBIC,Image.BICUBIC)), 
                        batch_tfms=batch_tfms,  
                        bs=bs, shuffle=shuffle
                        )
    return dls
     

def do_chunk(embs):
    step = 1000
    for chunk_start in range(0, embs.shape[0], step):
        chunk_end = min(chunk_start+step, len(embs))
        yield embs[chunk_start:chunk_end]

def f1_from_embs(embs, ys, display=False):
    target_matrix = ys[:,None]==ys[None,:]
    groups = [torch.where(t)[0].tolist() for t in target_matrix]
    dists, inds = get_nearest(embs, do_chunk(embs))
    pairs = sorted_pairs(dists, inds)[:len(embs)*10]
    scores =build_from_pairs(pairs, groups, display)
    return max(scores)


def get_nearest(embs, emb_chunks, K=None, sorted=True):
    if K is None:
        K = min(50, len(embs))
    distances = []
    indices = []
    for chunk in emb_chunks:
        sim = embs @ chunk.T
        top_vals, top_inds = sim.topk(K, dim=0, sorted=sorted)
        distances.append(top_vals.T)
        indices.append(top_inds.T)
    return torch.cat(distances), torch.cat(indices)

def sorted_pairs(distances, indices):
    triplets = []
    n= len(distances)
    for x in range(n):
        used=set()
        for ind, dist in zip(indices[x].tolist(), distances[x].tolist()):
            if not ind in used:
                triplets.append((x, ind, dist))
                used.add(ind)
    return sorted(triplets, key=lambda x: -x[2])

def f1(tp, fp, num_tar):
    return 2 * tp / (tp+fp+num_tar)

def build_from_pairs(pairs, target, display = True):
    score =0
    tp = [0]*len(target)
    fp = [0]*len(target)
    scores=[]
    vs=[]
    group_sizes = [len(x) for x in target]
    for x, y, v in pairs:
        group_size = group_sizes[x]
        score -= f1(tp[x], fp[x], group_size)
        if y in target[x]: tp[x] +=1
        else: fp[x] +=1
        score += f1(tp[x], fp[x], group_size) 
        scores.append(score / len(target))
        vs.append(v)
    if display:
        plt.plot(scores)
        am =torch.tensor(scores).argmax()
        print(f'{scores[am]:.3f} at {am/len(target)} pairs or {vs[am]:.3f} threshold')
    return scores


def split_2way(model):
    return L(params(model.body) + params(model.after_conv),
            params(model.classifier))

def modules_params(modules):
    return list(itertools.chain(*modules.map(params)))

def split_nfnet(model):
    body =model.body 
    children = L(body.children())
    group1 =children[:1]
    group2 = children[1:]
    group3 = L([model.after_conv,model.classifier])
    return [modules_params(g) for g in [group1,group2,group3]]

def save_without_classifier(model, fname):
    model.classifier = None
    torch.save(model.state_dict(), fname)

class ConfigClass():
    def toDict(self):
        return {k:self.__getattribute__(k) for k in dir(self) if k[:2]!='__' and not inspect.isroutine(self.__getattribute__(k))}
    
    def fromDict(d):
        res = ConfigClass()
        for k,v in d.items():
            res.__setattr__(k,v)
        return res

    def __repr__(self):
        return str(self.toDict())
        
class ArcFaceLoss(Module):
    y_int=True
    def __init__(self, m: float = 0.5, s: int =30, output_classes: int =11014, weight=None, reduction='mean'):
        self.m=m 
        self.s=s 
        self.output_classes=output_classes
        self.weight=weight 
        self.reduction=reduction
    
    def forward(self, cosine, targ):
        cosine = cosine.clip(-1+1e-7, 1-1e-7) 
        arcosine = cosine.arccos()
        arcosine += F.one_hot(targ, num_classes = self.output_classes) * self.m
        cosine2 = arcosine.cos()
        cosine2 *= self.s
        return F.cross_entropy(cosine2, targ)

    def set_margin(self, new_m):
        self.m = new_m

class ArcFaceLossFlat(BaseLoss):
    y_int = True 
    def __init__(self, *args, m: float = 0.5, s: int =30, output_classes: int =11014, weight=None, reduction='mean', axis=-1):
        super().__init__(ArcFaceLoss, *args, m=m, s=s, output_classes=output_classes, weight=weight, reduction=reduction)
        
    def decodes(self, x): 
        return x.argmax(dim=self.axis)
    
    def activation(self, x): 
        return F.softmax(x, dim=self.axis)
    
    def set_margin(self, new_m):
        self.func.set_margin(new_m)

class ArcFaceClassifier(nn.Module):
    def __init__(self, in_features=512, output_classes=11014):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_features, output_classes))
        nn.init.kaiming_uniform_(self.W)
    def forward(self, x):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        return x_norm @ W_norm


class MarginScheduler(Callback):
    def __init__(self, start_m, end_m):
        self.start_m = start_m
        self.end_m = end_m

    def after_create(self):
        try:
            if self.learn.loss_func is None : 
                raise ValueError
            if getattr(self.learn.loss_func, 'm', False) is False :
                raise ValueError
        except ValueError as e:
            print( e," Loss function not defined or loss function has no margin attribute.")

    def before_fit(self):
        if self.learn.n_epoch > 1:
            step = (self.end_m - self.start_m) / (self.learn.n_epoch -1)
            self.margins = np.arange(self.start_m, self.end_m + step,step)
        else:
            self.learn.loss_func.set_margin(self.end_m)

    def before_epoch(self):
        if self.learn.epoch > 0:
            self.learn.loss_func.set_margin(self.margins[self.learn.epoch])


class F1FromEmbs(Callback):
    def after_pred(self):
        if not self.training:
            self.embs.append(self.learn.pred[1])
            self.ys.append(self.learn.yb[0])
            self.learn.pred = self.learn.pred[0]
    def before_validate(self):
        self.ys = []
        self.embs = []
        self.model.outputEmbs = True
    def before_train(self):
        self.model.outputEmbs = False
    def after_validate(self):
        embs = torch.cat(self.embs)
        embs = F.normalize(embs)
        ys = torch.cat(self.ys)
        score = f1_from_embs(embs,ys)
        self.learn.metrics[0].val = score

class F1EmbedMetric(Metric):
    val =0.0
    @property
    def value(self):
        return self.val
    
    @property
    def name(self): 
        return 'F1 embeddings'
    
#Taken from https://www.kaggle.com/c/shopee-product-matching/discussion/233605#1278984
def string_escape(s, encoding='utf-8'):
    return s.encode('latin1').decode('unicode-escape').encode('latin1').decode(encoding)

class TitleTransform(Transform):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
               
    def encodes(self, row):
        text = row.title
        text=string_escape(text)
        encodings = self.tokenizer(text, padding = 'max_length', max_length=100, truncation=True,return_tensors='pt')
        keys =['input_ids', 'attention_mask']
        return tuple(encodings[key].squeeze() for key in keys)

def get_text_dls(df, tokenizer, bs):
    tfm = TitleTransform(tokenizer)

    data_block = DataBlock(
        blocks = (TransformBlock(type_tfms=tfm), 
                  CategoryBlock(vocab=df.label_group.to_list())),
        splitter=ColSplitter(),
        get_y=ColReader('label_group'),
        )
    return  data_block.dataloaders(df, bs=bs)

class NoLoss(Module):
    def __init__(self, weight=None,  reduction='mean'):
        self.weight=weight 
        self.reduction=reduction
    def forward(self, x, y):
        return 0

class GetEmbs(Callback):
    def after_pred(self):
        if not self.training:
            self.learn.model.collected = torch.cat([self.learn.model.collected,self.learn.pred],0)
            self.learn.yb = tuple()

class ModelGetEmbeddingWrapper(nn.Module):
    def __init__(self, model, embedding_size):
        super().__init__()
        self.model = model
        self.collected = torch.empty((0,embedding_size)).cuda()
    def forward(self, x):
        return self.model(x)

def get_embeddings(model,dls, embedding_size=1024):
    learn = Learner(dls= dls, model=ModelGetEmbeddingWrapper(model, embedding_size=embedding_size), loss_func=NoLoss(),cbs=GetEmbs())
    learn.validate()
    return learn.model.collected