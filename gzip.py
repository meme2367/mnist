
# coding: utf-8

# In[1]:

#mnist4개파일 내려받고 GZip 압축 해제하는 프로그램
import urllib.request as req
import gzip,os,os.path


# In[3]:

savepath = './mnist'
baseurl = "http://yann.lecun.com/exdb/mnist"
files = ["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz","t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"]


# In[5]:

#다운로드
if not os.path.exists(savepath):os.mkdir(savepath)
for f in files:
    url = baseurl + "/" + f
    loc = savepath + "/" + f
    print("download:",url)
    #urlretrieve(url,savename) 파일 저장
    if not os.path.exists(loc):req.urlretrieve(url,loc)
        


# In[6]:

#gzip압축해제
for f in files:
    gz_file = savepath + "/" +f
    #gz_files = ./mnist/train-images-idx3-ubyte.gz
    #raw_files = ./mnist/tr./mnist/train-images-idx3-ubyteain-images-idx3-ubyte
    raw_file = savepath + "/" + f.replace(".gz","")
    with gzip.open(gz_file,"rb") as fp:
        body = fp.read()
        #gz_file(./mnist/train-images-idx3-ubyte.gz)을 만들어서 읽음
        with gzip.open(raw_file,"wb") as w:
            #raw_file(./mnist/train-images-idx3-ubyte)을만들어서
            #./gz을 쓴다(압축해제)
            w.write(body)
print("ok")

