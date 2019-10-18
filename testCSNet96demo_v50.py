
import os,sys,glob,math
from time import time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from dwCSNet_model_v50 import CSNet
from csutils.pytorch_msssim import ssim
from csutils.cseval_metrics import compute_NRMSE

block_size =32;
dtype = torch.float32

def createDir(imgn,dirname,CS_ratio):
    img_path = os.path.dirname(imgn)        
    img_path = os.path.abspath(os.path.join(img_path, "..")) + dirname
    img_rec_path = "%s_rec_%s" % (img_path,CS_ratio)
    isExists=os.path.exists(img_rec_path)
    if not isExists:
        os.makedirs(img_rec_path)
        return img_rec_path
    else:
        return img_rec_path
    
# 输入参数：未正则化的image数据[0,255]
def psnrISTA(img_rec, img_orig):
    img_rec=img_rec.astype(np.float32)
    img_orig=img_orig.astype(np.float32)
    mse = np.mean((img_rec - img_orig) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 输入参数：正则化后的image数据[0,255]/255.0
def psnr(recovered, original):
    recovered=recovered.astype(np.float32)
    original=original.astype(np.float32)    
    recovered = torch.from_numpy(recovered)
    original = torch.from_numpy(original)
    
    mse = F.mse_loss(recovered, original)
    if mse == 0:
        return 100
    psnr = 10 * np.log10(1 / mse.item())
    return psnr
 
# 输入参数：未来正则化的image数据[0,255]
# https://github.com/hvcl/RefineGAN
def psnr_RefineGAN(prediction, ground_truth, maxp=255.):
    """`Peek Signal to Noise Ratio 
        PSNR = 20 \cdot \log_{10}(MAX_p) - 10 \cdot \log_{10}(MSE)
    Args:
        maxp: maximum possible pixel value of the image (255 in in 8bit images)
    Returns:
        A scalar representing the PSNR.
    """
    prediction   = np.abs(prediction)
    ground_truth = np.abs(ground_truth)

    mse = np.mean(np.square(prediction - ground_truth))
    if maxp is None:
        psnr = np.multiply(np.log10(mse), -10.)
    else:
        maxp = float(maxp)
        psnr = np.multiply(np.log10(mse+1e-6), -10.)
        psnr = np.add(np.multiply(20., np.log10(maxp)), psnr)

    return psnr
    
def CSNetRGBRec(model, img_padding, device, img_orig, channels_Num):
#     [row, col,channels_Num] = img_orig.shape 
    row = img_orig.shape[0]
    col = img_orig.shape[1]
     
    RGB_rec = []   
    rec_PSNR =0.0
    ssim_val =0.0 
    nrmse_val = 0.0 
    ysize = 0.0
    timerec =0.0
    with torch.no_grad():
        img_padding = img_padding.to(device=device, dtype=dtype)
        if channels_Num==1:
            statT = time()
            inputimgcp = img_padding.view(1,1,img_padding.size()[0],img_padding.size()[1])   
            img_rec, outcsy, out_initrec = model(inputimgcp)  
            timerec = time() - statT
            ysize = sys.getsizeof(outcsy.cpu().numpy())
                        
            img_recinit = out_initrec.view(out_initrec.size()[2],out_initrec.size()[3])
            img_recinit = img_recinit.cpu().numpy()
            # 必须使用np.clip()函数防止最大值超过255导致高亮区域图像像素显示错误
            out_initrec = np.clip(img_recinit, 0, 255).astype(np.uint8) 
            
            # Use Tensor.cpu() to copy the tensor to host memory first
            img_recch = img_rec.view(img_padding.size()[0],img_padding.size()[1])
            img_recch = img_recch.cpu().numpy()
            # 必须使用np.clip()函数防止最大值超过255导致高亮区域图像像素显示错误
            RGB_rec = np.clip(img_recch[:row, :col], 0, 255).astype(np.uint8)  # must be converted into np.uint8 then show correct images
#             rec_PSNR = psnr(RGB_rec/255.0, img_orig/255.0)
            rec_PSNR = psnr_RefineGAN(RGB_rec, img_orig)
            nrmse_val = compute_NRMSE(RGB_rec, img_orig)
            
            img_rect = torch.from_numpy(RGB_rec.astype(np.float32))
            img_rect = img_rect.view(1,1,row, col)
            img_orig = torch.from_numpy(img_orig.astype(np.float32))
            img_orig = img_orig.view(1,1,row, col)
            ssim_val = ssim(img_rect, img_orig)

        else:
            for channel_no in range(channels_Num):
                inputimgcp = img_padding[:,:,channel_no].view(1,1,img_padding.size()[0],img_padding.size()[1])
                img_recch, outcsy, _ = model(inputimgcp)  
                ysize = ysize + sys.getsizeof(outcsy.cpu().numpy())            
                img_recch = img_recch.view(img_padding.size()[0],img_padding.size()[1])
                img_recch = img_recch.cpu().numpy()
#                 imgf_x = np.clip(img_recch[:row, :col]*255.0, 0, 255).astype(np.uint8)
                imgf_x = np.clip(img_recch[:row, :col], 0, 255).astype(np.uint8)
                imgrec_x = Image.fromarray(imgf_x)
                RGB_rec.append(imgrec_x)                 
                rec_PSNR = rec_PSNR + psnr(imgf_x/255.0, img_orig[:,:,channel_no]/255.0)
                nrmse_val = compute_NRMSE(imgf_x, img_orig[:,:,channel_no])
                
                img_rect = torch.from_numpy(imgf_x.astype(np.float32))
                img_rect = img_rect.view(1,1,row, col)
                img_origt = torch.from_numpy(img_orig[:,:,channel_no].astype(np.float32))
                img_origt = img_origt.view(1,1,row, col)
                ssim_val = ssim_val + ssim(img_rect, img_origt)
                    
    if channels_Num==3:
            RGBimg_rec=Image.merge("RGB", (RGB_rec[0],RGB_rec[1],RGB_rec[2]))
            rec_PSNR = rec_PSNR /3.0
            ssim_val = ssim_val/3.0
            nrmse_val = nrmse_val/3.0   
    elif channels_Num==1:
#             RGB_rec = (RGB_rec*255.0).astype(np.uint8)  # must be converted into np.uint8 then show correct images
            RGBimg_rec = Image.fromarray(RGB_rec, 'L')
            out_initrec = Image.fromarray(out_initrec,"L")
            out_initrec.save("fp.bmp")
            rec_PSNR = rec_PSNR
            nrmse_val = nrmse_val   
  
#     RGBimg_rec.show()    
#     plt.imshow(RGBimg_rec)
#     plt.show()               
    return RGBimg_rec,rec_PSNR, ssim_val, nrmse_val, timerec
    
    
def ReadImgTensor(imgpath):    
    img_rgb = Image.open(imgpath);
#    https://www.aiuai.cn/aifarm472.html 
#     img_rgb = img_rgb.convert('I');  # to gray images
#     img = np.array(img_rgb, dtype=np.int32) 
#     img_rgb.show()
    img = np.array(img_rgb, dtype=np.uint8)   
    img_bsize = sys.getsizeof(img);  
#     [row, col, channels_Num] = img.shape
    channels_Num = len(img_rgb.split())
    row = img.shape[0]
    col = img.shape[1]
    
    if np.mod(row,block_size)==0:
        row_pad=0
    else:    
        row_pad = block_size-np.mod(row,block_size)
    
    if np.mod(col,block_size)==0:
        col_pad = 0
    else:        
        col_pad = block_size-np.mod(col,block_size)
    row_new = row + row_pad
    col_new = col + col_pad
    
    img_pad = []  
    if channels_Num==1:        
        imgorg=img[:,:]
        Ipadc = np.concatenate((imgorg, np.zeros([row, col_pad],dtype=np.uint8)), axis=1)
        Ipadc = np.concatenate((Ipadc, np.zeros([row_pad, col_new],dtype=np.uint8)), axis=0)   
        img_x = torch.from_numpy(Ipadc)
        img_padding = img_x.view(row_new, col_new,1)        
#         imgp = img_padding.cpu().view(img_padding.size()[0],img_padding.size()[1]).numpy()
#         imgp = (imgp*255.0).astype(np.uint8)  # must be converted into np.uint8 then show correct images
#         imgpad = Image.fromarray(imgp, 'L')
#         imgpad.show()        
    else:
        for channel_no in range(channels_Num):
#             print("channel no ====%d"%(channel_no))
            imgorg=img[:,:,channel_no]
            Ipadc = np.concatenate((imgorg, np.zeros([row, col_pad],dtype=np.uint8)), axis=1)
            Ipadc = np.concatenate((Ipadc, np.zeros([row_pad, col_new],dtype=np.uint8)), axis=0) 
#             img_x = torch.from_numpy(Ipadc/255.0)
            img_x = torch.from_numpy(Ipadc)
            img_pad.append(img_x.view(row_new, col_new,1))
        img_padding = torch.cat(img_pad,dim=2)
    
    return img, channels_Num, img_bsize, img_padding

def CSNetTesting(filepaths,model_sdict, CS_ratio, save_img=False):
    # set up device
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
#     device = torch.device('cpu')
    print('using device:', device)
     
#   To load the model, use the following code:
    model = CSNet(CS_ratio)
    model.load_state_dict(torch.load(model_sdict))
    model.eval()  # ensure the model is in evaluation mode
    model.to(device)
    print('trained model loaded')    

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([ImgNum], dtype=np.float32) 
    SSIM_All = np.zeros([ImgNum], dtype=np.float32) 
    nrmse_All = np.zeros([ImgNum], dtype=np.float32)
    MCRy = np.zeros([ImgNum], dtype=np.float32) 
    img_rec_path = createDir(filepaths[0],'/CSNet96',(CS_ratio)[2:4]);
    
    statT = time()
    for img_no in range(ImgNum): 
        imgName = filepaths[img_no]    
        img_orig, channels_Num, img_bsize, img_padding= ReadImgTensor(imgName)                 
        RGBimg_rec, rec_PSNR, ssim_val,nrmse_Val,rectime =  CSNetRGBRec(model, img_padding, device, img_orig, channels_Num) 
        PSNR_All[img_no] = rec_PSNR  
        SSIM_All[img_no] = ssim_val 
        nrmse_All[img_no] = nrmse_Val 
        MCRy[img_no] = rectime
        print("Image %s, PSNR= %.6f, SSIM=%0.3f, NRMSE=%0.3f, mCR= %0.3f" % (imgName, rec_PSNR, ssim_val, nrmse_Val,MCRy[img_no]))
        
        img_name = os.path.split(imgName)[-1]        
        img_rec_name = "%s/%s" % (img_rec_path, img_name)  
        if save_img==True:  
            RGBimg_rec.save(img_rec_name, quality=100, subsampling=0) 
        print("Rec_image save to",img_rec_name) 
     
    #-------------------------------------------------
    print("-----------------------")     
    print("---=Average Time(s)=---", (time()-statT)/ImgNum, "---=Average Speed(image/s)=---", ImgNum/(time()-statT))
    output_data = "CS_ratio= %.2f , AvgPSNR is %.2f dB, AvgSSIM is %.3f, AvgNRMSE is %.3f, rectime is %.3f \n" % (float(CS_ratio), np.mean(PSNR_All), np.mean(SSIM_All), np.mean(nrmse_All), np.mean(MCRy))
    print(output_data)
    output_psnr = "min= %.3f, Q1= %.3f, median= %.3f, Q3= %.3f, max= %.3f, std=%.3f" % (np.min(PSNR_All), np.percentile(PSNR_All,25), np.median(PSNR_All), np.percentile(PSNR_All,75), np.max(PSNR_All), np.std(PSNR_All))
    output_ssim = "min= %.3f, Q1= %.3f, median= %.3f, Q3= %.3f, max= %.3f, std=%.3f" % (np.min(SSIM_All), np.percentile(SSIM_All,25), np.median(SSIM_All), np.percentile(SSIM_All,75), np.max(SSIM_All),np.std(SSIM_All))
    output_nrmse = "min= %.3f, Q1= %.3f, median= %.3f, Q3= %.3f, max= %.3f, std=%.3f" % (np.min(nrmse_All), np.percentile(nrmse_All,25), np.median(nrmse_All), np.percentile(nrmse_All,75), np.max(nrmse_All),np.std(nrmse_All))
    print("PSNR: ",output_psnr)
    print("SSIM: ",output_ssim)
    print("NRMSE: ",output_nrmse)      

if __name__ == '__main__':   

    path_dataset = "./Test_Image/brain_valid" 
    filepaths = glob.glob(path_dataset + '/*.png')  
#     filepaths = glob.glob(path_dataset + '/040.png')  
   
    csrate = '0.30'    
    CSNetmodel_sdict = "CSNet96_%s.pt" % (csrate)[2:4]
    CSNetTesting(filepaths,CSNetmodel_sdict,csrate,True)

