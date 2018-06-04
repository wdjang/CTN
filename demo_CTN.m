clear

caffe.reset_all();

warning('off','all');

addpath('./GrabCut');
addpath('./flow-code-matlab');

param.db_path = './data/JPEGImages';
param.gt_path = './data/Annotations';
param.flow_path = './data/optical_flow';
param.out_path = './results';

param.beta = 0.3;
param.fgk = 10;
param.bgk = 10;
param.maxIter = 10;
param.diffThreshold = 0.001;
param.G = 25;

param.rho = 0.3;
param.sigma = 0.9;

%% 
% Config
config.use_gpu = true;
config.gpu_id = 0;
config.net_model = './model/CTN_deploy.prototxt';
config.net_weight = './model/CTN_iter_282816.caffemodel';    % ssvos_iter_282816

% Add MatCaffe Path
matcaffe_path = './caffe-upm/matlab/';
addpath(matcaffe_path)

% Caffe Setup
if config.use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(config.gpu_id);
else
  caffe.set_mode_cpu();
end

net = caffe.Net(config.net_model, 'test');
net.copy_from(config.net_weight);

param.datac_sz = net.blobs('datac').shape();
param.datac_sz(4) = 1;
param.propc_sz = net.blobs('propc').shape();
param.propc_sz(4) = 1;
param.propb_sz = net.blobs('propb').shape();
param.propb_sz(4) = 1;

%%
db_list = dir(param.db_path);
db_list = db_list(3:end);

test_list = 1:length(db_list);

for db_id = test_list
    
    fprintf('%s... ',db_list(db_id).name);
    stt_tic = tic;
    seg_track = CTN(param,db_list(db_id).name,net);
   
    fprintf('Running time: %.3f(spf)\n\n',toc(stt_tic)/length(seg_track));
    if ~exist(fullfile(param.out_path,db_list(db_id).name),'dir')
        mkdir(fullfile(param.out_path,db_list(db_id).name));
    end
    
    for f_id = 1:length(seg_track)
        imwrite(seg_track{f_id},fullfile(param.out_path,db_list(db_id).name,sprintf('%05d.png',f_id-1)));
    end
    
end

caffe.reset_all();
