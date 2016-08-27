

%% ��ջ�������
close all;
clear;
clc;
format compact;
%% ������ȡ

% �����������wine,���а���������Ϊclassnumber = 3,wine:178*13�ľ���,wine_labes:178*1��������
load mix3.mat;
load mixlabel.mat;
classnumber=3;
% �����������ݵ�box���ӻ�ͼ
% figure;
% boxplot(wine,'orientation','horizontal','labels',categories);
% title('wine���ݵ�box���ӻ�ͼ','FontSize',12);
% xlabel('����ֵ','FontSize',12);
% grid on;

% �����������ݵķ�ά���ӻ�ͼ
% figure
% subplot(3,5,1);
% hold on
% for run = 1:178
%     plot(run,wine_labels(run),'*');
% end
% xlabel('����','FontSize',10);
% ylabel('����ǩ','FontSize',10);
% title('class','FontSize',10);
% for run = 2:14
%     subplot(3,5,run);
%     hold on;
%     str = ['attrib ',num2str(run-1)];
%     for i = 1:178
%         plot(i,wine(i,run-1),'*');
%     end
%     xlabel('����','FontSize',10);
%     ylabel('����ֵ','FontSize',10);
%     title(str,'FontSize',10);
% end

% ѡ��ѵ�����Ͳ��Լ�

% ����һ���1-150,�ڶ����301-450,�������601-750��Ϊѵ����
train_wine = [mix3(1:150,:);mix3(301:450,:);mix3(601:750,:)];
% ��Ӧ��ѵ�����ı�ǩҲҪ�������
train_wine_labels = [mixlabel(1:150);mixlabel(301:450);mixlabel(601:750)];
% ����һ���151-300,�ڶ����450-600,�������751-900��Ϊ���Լ�
test_wine = [mix3(151:300,:);mix3(450:600,:);mix3(751:900,:)];
% ��Ӧ�Ĳ��Լ��ı�ǩҲҪ�������
test_wine_labels = [mixlabel(151:300);mixlabel(450:600);mixlabel(751:900)];

% %% ����Ԥ����
% % ����Ԥ����,��ѵ�����Ͳ��Լ���һ����[0,1]����
% 
% [mtrain,ntrain] = size(train_wine);
% [mtest,ntest] = size(test_wine);
% 
% dataset = [train_wine;test_wine];
% % mapminmaxΪMATLAB�Դ��Ĺ�һ������
% [dataset_scale,ps] = mapminmax(dataset',0,1);
% dataset_scale = dataset_scale';
% 
% train_wine = dataset_scale(1:mtrain,:);
% test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% SVM����ѵ��
model = svmtrain(train_wine_labels, train_wine, '-c 2 -g 1');

%% SVM����Ԥ��
%[predict_label, accuracy] = svmpredict(test_wine_labels, test_wine, model);
[predict_label, accuracy,decision_values] = svmpredict(test_wine_labels, test_wine, model);

%% �������

% ���Լ���ʵ�ʷ����Ԥ�����ͼ

figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_label,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;