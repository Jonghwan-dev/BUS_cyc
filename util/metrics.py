from skimage import metrics

'''
                #############val###############
            with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                NCC = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A_valid.copy_(batch['A']))
                    real_B = Variable(self.input_B_valid.copy_(batch['B']))
                    fake_B, aux_out = self.netG_A2B(real_A)

                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    real_B = real_B.detach().cpu().numpy().squeeze()
                    dr = 2.0
                    mae = self.MAE(fake_B,real_B)
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = metrics.structural_similarity(fake_B,real_B, data_range=dr, multichannel=False)
                    ncc = self.normxcorr2(fake_B, real_B, mode="valid")[0,0]

                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    NCC += ncc
                    
                    num += 1
                
                self.log.info('Val MAE: {}'.format(MAE/num))
                self.log.info('Val PSNR: {}'.format(PSNR/num))
                self.log.info('Val SSIM: {}'.format(SSIM/num))
                self.log.info('Val NCC: {}'.format(NCC/num))

                score = (0.1*PSNR + SSIM + NCC)/num
                if score > best_score:
                    best_score = score
                    np.save(os.path.join(self.config['save_root'], 'best_score.npy'), best_score)
                    self.log.info('best score: {}'.format(best_score))
                    torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'best_netG_A2B.pth')
                    torch.save(self.netG_B2A.state_dict(), self.config['save_root'] + 'best_netG_B2A.pth')
                    torch.save(self.netD_A.state_dict(), self.config['save_root'] + 'best_netD_A.pth')
                    torch.save(self.netD_B.state_dict(), self.config['save_root'] + 'best_netD_B.pth')
                    
'''                         
    
    def PSNR(self,fake,real):
       x,y = np.where(real!= -1)# Exclude background
       mse = np.mean(((fake[x,y]+1)/2. - (real[x,y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
    def MAE(self,fake,real):
        x,y = np.where(real!= -1)  # Exclude background
        mae = np.abs(fake[x,y]-real[x,y]).mean()
        return mae/2     #from (-1,1) normaliz  to (0,1)
    
    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 

    def normxcorr2(self, template, image, mode="full"):
        """
        Input arrays should be floating point numbers.
        :param template: N-D array, of template or filter you are using for cross-correlation.
        Must be less or equal dimensions to image.
        Length of each dimension must be less than length of image.
        :param image: N-D array
        :param mode: Options, "full", "valid", "same"
        full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
        Output size will be image size + 1/2 template size in each dimension.
        valid: The output consists only of those elements that do not rely on the zero-padding.
        same: The output is the same size as image, centered with respect to the 'full' output.
        :return: N-D array of same dimensions as image. Size depends on mode parameter.
        """

        # If this happens, it is probably a mistake
        if np.ndim(template) > np.ndim(image) or \
                len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
            print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

        template = template - np.mean(template)
        image = image - np.mean(image)

        a1 = np.ones(template.shape)
        # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
        ar = np.flipud(np.fliplr(template))
        out = fftconvolve(image, ar.conj(), mode=mode)
        
        image = fftconvolve(np.square(image), a1, mode=mode) - \
                np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

        # Remove small machine precision errors after subtraction
        image[np.where(image < 0)] = 0

        template = np.sum(np.square(template))
        with np.errstate(divide='ignore',invalid='ignore'): 
            out = out / np.sqrt(image * template)

        # Remove any divisions by 0 or very close to 0
        out[np.where(np.logical_not(np.isfinite(out)))] = 0
        
        return out