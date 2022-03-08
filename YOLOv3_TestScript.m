%% YOLOv3 Test Script
% Fred liu 2021.11.29
% Create for medicial Image Deep Learning

%% Basic Setting(基礎設置)

% Load Model載入模型
%load('Yolov3_detector211202V1.mat')
load('Yolov3_detector220107_v1.mat')

% Choose DataSet 選擇資料集
datasetNum = 1;

% 模型輸入尺寸 yolov3-Tiny 416 yolov3-darknet53  608
networkInputSize = [416 416];
%networkInputSize = [608 608];
%% Load XML File & String Normalization(載入XML標記資訊 & 字串正規化)

switch datasetNum
    case 1
        % Test
        dsTest = fileDatastore('Test_Dataset\Test_Labels','ReadFcn',@readFcn);
        T_bbox= readall(dsTest);
        ds2 = fileDatastore('Test_Dataset\Test_Labels','ReadFcn',@readFcn2);
        dsTest_file= readall(ds2);
        Test_gTruth_labeler = table(dsTest_file,T_bbox);

        PathTest = 'D:\Fred\MATLAB_Project(customer)\2021\DL競賽\VGH_DATA\Test_Dataset\Test_Images';
        PathTest2 = [PathTest,'\'];
        TestImg = strcat(PathTest2,string(Test_gTruth_labeler.dsTest_file));

    case 2
        % Train
        dsTest = fileDatastore('Train_Dataset\Train_Labels','ReadFcn',@readFcn);
        T_bbox= readall(dsTest);
        ds2 = fileDatastore('Train_Dataset\Train_Labels','ReadFcn',@readFcn2);
        dsTest_file= readall(ds2);
        Test_gTruth_labeler = table(dsTest_file,T_bbox);

        PathTest = 'D:\Fred\MATLAB_Project(customer)\2021\DL競賽\VGH_DATA\Train_Dataset\Train_Images';
        PathTest2 = [PathTest,'\'];
        TestImg = strcat(PathTest2,string(Test_gTruth_labeler.dsTest_file));
end

%% Bulid Datastore(資料整合 建Datastore)

imdsTest = imageDatastore(TestImg);
bldsTest = boxLabelDatastore(Test_gTruth_labeler(:, 2:end));
testData = combine(imdsTest, bldsTest);
%% Evaluate Model(評估模型 檢測resize bbox)
testDataForEstimation = transform(testData, @(data)preprocessData(data, networkInputSize));
% Test image 測試影像
[data,info] = read(testDataForEstimation);

I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% YOLOv3 Pre-Processing(YOLOv3前處理)
preprocessedTrainingData = transform(testData, @(data)preprocess(yolov3Detector, data));
data = read(preprocessedTrainingData);

I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
reset(preprocessedTrainingData);
%% Detct(檢測)
results = detect(yolov3Detector,preprocessedTrainingData,'MiniBatchSize',8,'Threshold',0.001);

% Evaluate the object detector using Average Precision metric.
[ap,recall,precision] = evaluateDetectionPrecision(results,preprocessedTrainingData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
%% Test Single Image(測試單張影像)
% Read the datastore.
data = read(testData);

% Get the image
I = data{1};

[bboxes,scores,labels] = detect(yolov3Detector,I);

% Display the detections on image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);

figure
imshow(I)
%% Sub Support Function

function data = preprocessData(data, targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    % Convert an input image with single channel to 3 channels.
    if numel(imgSize) < 3 
        I = repmat(I,1,1,3);
    end
    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    
    data(ii, 1:2) = {I, bboxes};
end
end

