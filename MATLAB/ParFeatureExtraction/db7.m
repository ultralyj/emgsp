clear all; clc

func_list = ['getrmsfeat','getTDfeat','getmavfeat','getzcfeat','getsscfeat','getwlfeat','getarfeat','getiavfeat','getHISTfeat','getmDWTfeat'];
for i = 2:22
    clear feat_set
    E1 = load("E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E1_A1.mat");
    E2 = load("E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E2_A1.mat");
    emg = [E1.emg ;E2.emg];
    restimulus = [E1.restimulus ;E2.restimulus];
    rerepetition = [E1.rerepetition ;E2.rerepetition];
    
    for j = 1:10
        [feat, featStim, featRep] = ParFeatureExtractor(emg,restimulus,rerepetition,10^-5,400,20,'getrmsfeat');
        feat_set(:,:,j) = feat;
    end
    filename = "feature_S"+num2str(i)+".mat";
    save(filename,"feat_set","featStim","featRep")
end


