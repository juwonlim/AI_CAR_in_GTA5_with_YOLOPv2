#내가 추가한 모듈들
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
from loguru import logger
#from yolop.utils.visualize import vis
#from yolop.data.datasets.transform import ValTransform
import numpy as np
from yolopv2_inference.utils.utils import letterbox

#아래 내가 추가한 경로
weight_path = 'E:/gta5_project/AI_CAR_in_GTA5_with_YOLOPv2/yolopv2_inference/data/weights/yolopv2.pt' #이렇게 하면 리스트 형태로 전달됨,torch.jit.load는 리스트 대신 문자열 경로가필요함
sample_image = 'E:/gta5_project/AI_CAR_in_GTA5_with_YOLOPv2/sample_images/8.jpg'


import argparse
import time
from pathlib import Path
#import cv2
#import torch






# Conclude setting / general reprocessing / plots / metrices / datasets
#yolopv2_inference.utils.utils --> 내가 수정한 경로
from yolopv2_inference.utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages




def make_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default='weight_path', help='model.pt path(s)') --> 문법 오류
    parser.add_argument('--weights', nargs='+', type=str, default=[weight_path], help='model.pt path(s)') #nargs='+' 는 --weights가 여러 개의 값을 받을 수 있도록 리스트로 처리,그런데 실제로는 하나만 전달했기 때문에, 리스트 안에 단일 문자열이 들어 있는 형태
    #parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    #parser.add_argument('--source', type=str, default='sample_image', help='source')  # file/folder, 0 for webcam --> 문법 오류
    parser.add_argument('--source', type=str, default=sample_image, help='source') 
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser


def detect():
    # setting and directories
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride =32
    model  = torch.jit.load(weights)  
    
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)
    

    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
          
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


#############################위는 demo.py의 핵심원본 코드들 , 건드리지말것#########################



from preprocess_for_yolopv2 import grab_screen


def detect_realtime():
    # 초기 설정
    weights, imgsz = opt.weights, opt.img_size
    save_img = not opt.nosave
    stride = 32

    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    #save_dir.mkdir(parents=True, exist_ok=True)

    # 성능 측정기
    #inf_time, waste_time, nms_time = AverageMeter(), AverageMeter(), AverageMeter()

    # 모델 로드
    #model = torch.jit.load(weights) #torch.jit.load()는 **문자열(str)**을 기대함 , model = torch.jit.load("E:/path/to/yolopv2.pt") --> OK (문자열)
                                     #weights는 리스트 → 에러: list has no attribute 'read'
    model  = torch.jit.load(weights[0]) #AttributeError: 'list' object has no attribute 'read' (이는 torch.jit.load() 함수에 list 타입을 넘겼기 때문에 발생한 오류)
                                        #default=[weight_path] 로 설정하면 opt.weights는 list 타입이 됨
                                        #하지만 torch.jit.load()는 string 경로 하나만 받아야 함. 즉, torch.jit.load(opt.weights[0]) 이렇게 수정
                                        #이weights[0]으로 리스트의 첫 번째 요소(= 문자열 경로)를 꺼내야 함

    
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # warm-up
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    while True:
        # 1. Grab screen from GTA5
        #screen = grab_screen(region=None) #preprocess의 grab_screen함수가 def grab_screen(): 이렇게 인자 없이 시작되기에 region이란게 여기 있으면 안됨
        # 1. 화면 캡처
        screen = grab_screen()

        if screen is None or screen.size == 0:
            print("grab_screen() 실패: 빈 이미지가 반환됨")
            continue  # 다음 루프로 넘어가서 다시 시도

        if screen is None or screen.shape[0] == 0 or screen.shape[1] == 0:
            print("grab_screen()에서 올바르지 않은 이미지 크기 반환. 루프를 계속합니다.")
            continue

        # 2. 시각화용 이미지 (Segmentation 결과 오버레이용)
        visual_img = cv2.resize(screen, (1280, 720))  # 혹은 모델 출력 해상도에 맞춤
        # 3. 모델 입력용 이미지 (letterbox로 정규화)
        #input_img = letterbox(visual_img, imgsz, stride=32)[0]
        #input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        #input_img = np.ascontiguousarray(input_img)


        #im0 = cv2.resize(screen, (imgsz, imgsz))
        #input_img = cv2.resize(screen, (imgsz, imgsz)) #input_img.shape = (640, 640, 3)인데,color_mask.shape = (720, 1280) 이런 식으로 서로 크기가 안 맞음.
                                                        
        #img = letterbox(input_img, imgsz, stride=32)[0]
        model_input = letterbox(visual_img, imgsz, stride=32)[0]
        model_input = model_input[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        model_input = np.ascontiguousarray(model_input)

        img_tensor = torch.from_numpy(model_input).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 2. 추론
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img_tensor)
        t2 = time_synchronized()

        # 3. 후처리
        #tw1 = time_synchronized() # waste time 측정
        pred = split_for_trace_model(pred, anchor_grid)
        #tw2 = time_synchronized() #waste time 측정

        #t3 = time_synchronized() #성능측정에만 사용
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        #t4 = time_synchronized() #이것도 성능측정

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # 4. 결과 처리 및 시각화
        for i, det in enumerate(pred):
            #s = '%gx%g ' % img_tensor.shape[2:]  --> s 변수는 원래 print() 출력용 문자열,지금은 print(f'{s}Done. (...)에만 쓰이고 있으므로, 줄여도 무방
            if len(det):
                #det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], input_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    #plot_one_box(xyxy, im0, line_thickness=2)
                    plot_one_box(xyxy, input_img, line_thickness=2)

            #print(f'{s}Done. ({t2 - t1:.3f}s)') #위의 S변수 주석처리에 따라 이것도 주석
            print(f'Done. ({t2 - t1:.3f}s)')

            #show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True) #원본
            output_img = show_seg_result(input_img, (da_seg_mask, ll_seg_mask), is_demo=True) #utils.py파일에서 show_seg_result함수의 return값을 img로 정해둠, 리턴값을 output_img에서 받음
            cv2.namedWindow("YOLOPv2 Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOPv2 Result", 640, 360)
            cv2.moveWindow("YOLOPv2 Result", 600, 750)
            cv2.imshow("YOLOPv2 Result", output_img)


        # 5. ESC 키 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == 27:
            print("종료 명령 수신. 루프를 종료합니다.")
            break

    print('완료. 총 처리 시간: (%.3fs)' % (time.time() - t0))



#### 아래는 원본 코드지만 수정 가함#####
if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        #detect()
        detect_realtime()
            
