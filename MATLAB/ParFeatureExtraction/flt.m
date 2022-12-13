clf;clear;
%设置初值
f0=50;
Ts=0.001;
fs=1/Ts;
NLen=512;
n=0:NLen-1;
%陷波器的设计
apha=-2*cos(2*pi*f0*Ts);
beta=0.96;
b=[1 apha 1];
a=[1 apha*beta beta^2];
figure('color',[1 1 1]);
freqz(b,a,NLen,fs);%陷波器特性显示
x=sin(2*pi*50*n*Ts)+sin(2*pi*125*n*Ts);%原信号
y=dlsim(b,a,x);%陷波器滤波处理
%对信号进行频域变换。
xfft=fft(x,NLen);
xfft=xfft.*conj(xfft)/NLen;
y1=fft(y,NLen);
y2=y1.*conj(y1)/NLen;
figure(2);%滤除前后的信号对比。
subplot(2,2,1);plot(n,x);grid;
xlabel('Time (s)');ylabel('Amplitude');title('Input signal');
subplot(2,2,3);plot(n,y);grid;
xlabel('Time (s)');ylabel('Amplitude');title('Filter output');
subplot(2,2,2);plot(n*fs/NLen,xfft);axis([0 fs/2 min(xfft) max(xfft)]);grid;
xlabel('Frequency (Hz)');ylabel('Magnitude (dB)');title('Input signal');
subplot(2,2,4);plot(n*fs/NLen,y2);axis([0 fs/2 min(y2) max(y2)]);grid;
xlabel('Frequency (Hz)');ylabel('Magnitude (dB)');title('Filter output');