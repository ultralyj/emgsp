clear all; clc

func_list = ["getrmsfeat","getmavfeat","getzcfeat","getsscfeat","getwlfeat","getiavfeat",""];
for i = 2:2
    clear acc_feat gyro_feat mag_feat
    disp("loading E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E1_A1.mat")
    E1 = load("E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E1_A1.mat");
    disp("loading E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E2_A1.mat")
    E2 = load("E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E2_A1.mat");
    acc = [E1.acc ;E2.acc];
    gyro = [E1.gyro ;E2.gyro];
    mag = [E1.mag ; E2.mag];
    restimulus = [E1.restimulus ;E2.restimulus];
    rerepetition = [E1.rerepetition ;E2.rerepetition];
    
    for j = 1:1
        disp("get feature:" + func_list(j))
        [feat, featStim, featRep] = ParFeatureExtractor(acc,restimulus,rerepetition,10^-5,400,20,"getTDfeat");
        acc_feat(:,:,j) = feat;
        [feat, featStim, featRep] = ParFeatureExtractor(gyro,restimulus,rerepetition,10^-5,400,20,"getTDfeat");
        gyro_feat(:,:,j) = feat;
        [feat, featStim, featRep] = ParFeatureExtractor(mag,restimulus,rerepetition,10^-5,400,20,"getTDfeat");
        mag_feat(:,:,j) = feat;
    end
    filename = "IMU_feature_S"+num2str(i)+".mat";
    save(filename,"acc_feat","gyro_feat","mag_feat","featStim","featRep")
end


