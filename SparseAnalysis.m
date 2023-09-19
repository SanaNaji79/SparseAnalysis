%% 
clear all ;
%%
clc ;
%% load data
load 'C:\Users\Sana\OneDrive\Desktop\semester8\AdvanceNeurosience_ghazizadeh\homework\assignments\HW9_SanaAminnaji_98104722\IMAGES.mat' ;
load 'C:\Users\Sana\OneDrive\Desktop\semester8\AdvanceNeurosience_ghazizadeh\homework\assignments\HW9_SanaAminnaji_98104722\IMAGES_RAW.mat' ;
%% show data
figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    imagesc(IMAGES(: , : , i)) ;
    %title('Images') ;
end
%% show raw data
figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    imagesc(IMAGESr(: , : , i)) ;
    %title('Raw Images') ;
end

%% initialization
A = rand(64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
%% the data
load IMAGES ;
%% running code
sparsenet ;
%% Yale dataset
load 'C:\Users\Sana\Downloads\Yale_64x64.mat' ;

random = randi([1 165] , 1 , 10) ;
IMAGESrs = fea(random , :) ;
IMAGESrs = IMAGESrs' ;

figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    a = reshape(IMAGESrs(: , i) , [64 , 64]) ;
    imagesc(a) ;
end
N=64 ;
M=10 ;

[fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

for i=1:M
  image=reshape(IMAGESrs(: , i) , [N , N]);
  If=fft2(image);
  imagew=real(ifft2(If.*fftshift(filt)));
  IMAGES(:,i)=reshape(imagew,N^2,1);
end

IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));


figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    a = reshape(IMAGES(: , i) , [64 , 64]) ;
    imagesc(a) ;
end
%% MINST dataset 
load 'C:\Users\Sana\Downloads\HW3_Neuroscience\data.mat'
random = randi([1 5000] , 1 , 10) ;
IMAGESrs = X([100 , 701 , 1280 , 1511 , 2389 , 2840 , 3104 , 3912 , 4416 , 4681] , :) ;
IMAGESrs = IMAGESrs' ;

figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    a = reshape(IMAGESrs(: , i) , [20 , 20]) ;
    imagesc(a) ;
end
%% whitening
N=20 ;
M=10 ;
[fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

for i=1:M
  image=reshape(IMAGESrs(: , i) , [N , N]);
  If=fft2(image);
  imagew=real(ifft2(If.*fftshift(filt)));
  IMAGES(:,i)=reshape(imagew,N^2,1);
end

IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));


figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    a = reshape(IMAGES(: , i) , [N , N]) ;
    imagesc(a) ;
end
%% Caltech101 dataset
b = zeros(56400 , 10) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0005.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 1) = reshape(c(1:188 , :)  , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0015.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 2) = reshape(c(1:188 , :) , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0001.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 3) = reshape(c(1:188 , :)  , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0023.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 4) = reshape(c(1:188 , :)  , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0047.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 5) = reshape(c(1:188 , :)  , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0041.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 6) = reshape(c(1:188 , :)  , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0062.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 7) = reshape(c(1:188 , :) , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0021.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 8) = reshape(c(1:188 , :) , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0029.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 9) = reshape(c(1:188 , :) , [56400 , 1]) ;
a = imread('C:\Users\Sana\Downloads\caltech-101\caltech-101\101_ObjectCategories\101_ObjectCategories\elephant\image_0016.jpg') ;
c = im2double(rgb2gray(a)) ;
b(: , 10) = reshape(c(1:188 , :) , [56400 , 1]) ;
%%
IMAGESrs1 = b ;
IMAGESrs = zeros(188^2 , 10) ;
figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    a = reshape(IMAGESrs1(: , i) , [188 , 300]) ;
    a = a(: , 57:244) ;
    IMAGESrs(: , i) = reshape(a , [188^2 , 1]) ;
    imagesc(a) ;
end

%%
N=188 ;
M=10 ;
[fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

for i=1:M
  image=reshape(IMAGESrs(: , i) , [N , N]);
  If=fft2(image);
  imagew=real(ifft2(If.*fftshift(filt)));
  IMAGES(:,i)=reshape(imagew,N^2,1);
end

IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));


figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    a = reshape(IMAGES(: , i) , [N , N]) ;
    imagesc(a) ;
end
%% video
vid=VideoReader('C:\Users\Sana\Downloads\Data_HW9\BIRD.avi') ; 
numFrames = vid.NumberOfFrames;
n=numFrames;
x = zeros(288 , 288 , n) ;
for iFrame = 1:n
  frames = read(vid, iFrame);
  c = im2double(rgb2gray(frames)) ;
  x(: , : , iFrame) = c(: , 33:320) ;
end 
%%
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    imagesc(x(: , : , 11*i)) ;
    IMAGESrs(: , i) = reshape(x(: , : , 11*i) , 288*288 , 1) ;
end
N=288 ;
M=10 ;
[fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

for i=1:M
  image=reshape(IMAGESrs(: , i) , [N , N]);
  If=fft2(image);
  imagew=real(ifft2(If.*fftshift(filt)));
  IMAGES(:,i)=reshape(imagew,N^2,1);
end

IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));


figure 
for i = 1:10
    subplot(2 , 5 , i) ;
    colormap(gray) ;
    a = reshape(IMAGES(: , i) , [N , N]) ;
    imagesc(a) ;
end
%%
figure(3)
  subplot(211), bar(S_var), title('s variance')
  subplot(212), bar(sqrt(sum(A.*A))), title('basis norm (L2)')
%%
s_all = zeros(64*288 , 11) ;
figure
for k = 1:11
X1=zeros(L,288);
for i=1:288
    r=BUFF+ceil((image_size-sz-2*BUFF)*rand);
    c=BUFF+ceil((image_size-sz-2*BUFF)*rand);
    X1(:,i)=reshape(x(r:r+sz-1,c:c+sz-1 , k*10),L,1);
end
S=cgf_fitS(A,X1,noise_var,beta,sigma,tol);
s_all(: , k) = reshape(S , 64*288 , 1) ;
subplot(11 , 1 , k) ;
plot(s_all(: , k)) ;
title(['frame number = ' , num2str(k*10)]) ;
xlim([1 , 64*288]) ;
end
%%
figure
for k = 1:11
    figure(4)
    a = histogram(s_all(: , k) ,'BinWidth', 0.4 , 'Normalization' , 'probability') ;
    figure(5)
    plot(a.BinEdges(1:end-1) , a.Values) ;
    xlabel('a\i') ;
    ylabel('p(a\i)') ;
    title('the probability of coefficients for different times') ;
    hold on ;
    %legend(['frame number = ' , num2str(k*10)]) ;
end
%%
figure
for k = 1:11
    plot(s_all(: , k)) ;
    hold on ;
    xlim([1 , 64*288]) ;
end
