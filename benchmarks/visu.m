clear

load('../out_visu/out_pa.mat')
load('../out_visu/out_res.mat')

Pt_v  = Pt;  Pt_v(Phase~=1)=NaN;
Vn_v  = Vn;  Vn_v(Phase~=1)=NaN;
tII_v = zeros(size(Pt));
tII_v(2:end-1,2:end-1) = tII; tII_v(Phase~=1)=NaN;

figure(1),clf

subplot(311), pcolor(x2rot, y2rot, Pt_v),shading flat,colorbar,axis tight,title('Pressure')
subplot(312), pcolor(x2rot, y2rot, Vn_v),shading flat,colorbar, title('||V||')
subplot(313), pcolor(x2rot, y2rot, tII_v),shading flat,colorbar, title('\tau_{II}')

print('-dpng','-r200','../figs/arolla2.png')
