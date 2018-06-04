function seg_mask = grab_cut()

set(handles.Processbar,'Visible','on');
global fixedBG;
global im;
global CurrRes;
global PrevRes;
PrevRes = CurrRes;
imd = double(im);
Beta = handles.Beta_value;
k = handles.K_value;
G = 50;
maxIter = 10;
diffThreshold = 0.001;
% Double, Logical
L = GCAlgo(imd, fixedBG,k,G,maxIter, Beta, diffThreshold, handles.Processbar);
L = double(1 - L);