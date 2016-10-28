from numpy import *
import numpy as np
import struct
import pylab
 
def loadImageSet(filename):
	print "load image set",filename
	binfile= open(filename, 'rb')
	buffers = binfile.read()
 
	head = struct.unpack_from('>IIII' , buffers ,0)
	print "head,",head
 
	offset = struct.calcsize('>IIII')
	imgNum = head[1]
	width = head[2]
	height = head[3]
	#[60000]*28*28
	bits = imgNum * width * height
	bitsString = '>' + str(bits) + 'B' #like '>47040000B'
 
	imgs = struct.unpack_from(bitsString,buffers,offset)
 
	binfile.close()
	imgs = np.reshape(imgs,[imgNum,width*height])
	print "load imgs finished"
	return imgs
 
def loadLabelSet(filename):
 
	print "load label set",filename
	binfile = open(filename, 'rb')
	buffers = binfile.read()
 
	head = struct.unpack_from('>II' , buffers ,0)
	print "head,",head
	imgNum=head[1]
 
	offset = struct.calcsize('>II')
	numString = '>'+str(imgNum)+"B"
	labels = struct.unpack_from(numString , buffers , offset)
	binfile.close()
	labels = np.reshape(labels,[imgNum,1])
 
	print 'load label finished'
	return labels
 

#input mnist and process image,label 
imgs = loadImageSet("train-images.idx3-ubyte")
labels = loadLabelSet("train-labels.idx1-ubyte")


#get label is 6's image
count2=0
for num in range(60000):
    if(labels[num,0]==6):
        count2=count2+1        

#transform image to matrixe        
two=np.zeros((count2,784))    
i=0
for num in range(60000):
    if(labels[num,0]==6):
        for count in range(784):
            two[i,count]=imgs[num,count]
            count=count+1
        i=i+1

#SVD        
U,S,V = linalg.svd(two,full_matrices=False)     

U=U*S
choX=[]
choY=[]
X=[]
Y=[]
index=[]

#spread dot 
for i in range(count2):
    if(i%10==0):
        X.append(U[i,0])
        Y.append(U[i,1])
        if(i%500==0):
            index.append(i)
            choX.append(U[i,0])
            choY.append(U[i,1])
            
            
    
  
   

    
#show the images
pylab.figure(figsize=(8,6), dpi=80)
pylab.gray()
pylab.plot(X,Y,'k.')
pylab.plot(choX,choY,'ro')
pylab.savefig('dot.png')
for z in range(len(index)):
    print U[index[z],0],U[index[z],1],z
    count=0
    one=np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            one[i,j]=two[index[z],count]
            count=count+1
    pylab.imshow(one)
    pylab.savefig(str(z)+'fig.png',figsize=(8,6), dpi=80)

              




