clear all; clc

func_list = ["getrmsfeat","getmavfeat","getzcfeat","getsscfeat","getwlfeat","getiavfeat","getmDWTfeat"];
for i = 21:21
    clear feat_set 
    disp("loading E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E1_A1.mat")
    E1 = load("E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E1_A1.mat");
    disp("loading E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E2_A1.mat")
    E2 = load("E:db7\Subject_"+num2str(i)+"\S"+num2str(i)+"_E2_A1.mat");
    emg = [E1.emg ;E2.emg];
    restimulus = [E1.restimulus ;E2.restimulus];
    rerepetition = [E1.rerepetition ;E2.rerepetition];
    
    for j = 1:1
        disp("get feature:" + func_list(j))
        [feat, featStim, featRep] = ParFeatureExtractor(emg,restimulus,rerepetition,10^-5,400,20,"getmDWTfeat");
        feat_set(:,:,j) = feat;
    end
    filename = "DWTfeature_S"+num2str(i)+".mat";
    save(filename,"feat_set","featStim","featRep")
end


