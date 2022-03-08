%% test
addpath('D:\Fred\MATLAB_Project(customer)\2021\DL競賽\VGH_DATA\Train_Dataset\Train_Labels')

xml_01 = 'export-screen-2019-Dec-02_0923.xml';
type(xml_01)
DOMnode = xmlread(xml_01);

xmltxt = readstruct(xml_01);

objectPos = xmltxt.object;
objectPosBbox = bboxTran(xmltxt.object(1:end).bndbox);
objectPosBbox2 = bboxTran(xmltxt.object(2).bndbox);
objectFile = xmltxt.filename;
%% auto labeler


%% BoundingBox 轉換 Helpe Function
% function boundingBox = bboxTran(PosBbox)
% if isstruct(PosBbox)
%     if all(isfield(PosBbox,{'xmin','ymin','xmax','ymax'}))
%         H = PosBbox.ymin - PosBox.ymax;
%         W = PosBbox.xmin - PosNox.xmax;
%         boundingBox = [PosBbox.xmin,PosBbox.ymin,W,H];
%     end
% end
% end


