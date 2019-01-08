package num;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.net.UnknownHostException;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.imageio.ImageIO;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import sun.misc.BASE64Decoder;

public class NumServlet extends HttpServlet {
	
	//处理页面上的请求
	public void service(HttpServletRequest req,HttpServletResponse resp) {
		System.out.println(req.getRemoteAddr());
		String strMethod = req.getParameter("method");
		System.out.println(strMethod);
		
		String strJson = "";
		if(strMethod.equalsIgnoreCase("predictImage")) {
			strJson = predictImage(req,resp);
		}
		
		try {
			resp.getWriter().write(strJson);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//
	public String predictImage(HttpServletRequest req,HttpServletResponse resp) {		
		String strJson = "";
		
		//获取从浏览器中发送过来的base64编码的图片，
		String strImageBase64 = req.getParameter("image");		
		int pos = strImageBase64.indexOf("base64")+6;
		strImageBase64 = strImageBase64.substring(pos+1,strImageBase64.length()-1);		
		
		//将base64编码的图片数据保存为jpg格式
		SimpleDateFormat df = new SimpleDateFormat("yyyyMMddHHmmss");
		String strName = (df.format(new Date()));		
		String strImageFilePath = req.getServletContext().getRealPath("/")+"temp/"+strName + ".jpg";
		ConverBase64ToImage(strImageBase64,strImageFilePath);
		
		//将jpg图片缩放到mnist数据集中图片一致的大小，28x28
		scaleToMnist(strImageFilePath, strImageFilePath);
		
		//图片灰度化处理
		imageGrayScale(strImageFilePath);
		
		//通过tcp发送给python预测程序，进行cnn计算
		strJson = computeProbabilityByCNN();
		
		return strJson;
	} 	

	//通过tcp发送给python预测程序，进行cnn计算
	public String computeProbabilityByCNN() {    	
		int ret_predict = -1;       
       	Socket socket;
		try {
			socket = new Socket("127.0.0.1", 50008);
	       	DataInputStream input = new DataInputStream(socket.getInputStream());   
	       	ret_predict = input.readByte(); 
	       	socket.close();
		} catch (UnknownHostException e) {			
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	        
        String strRet = "";
        if(ret_predict != -1) {
        	strRet = String.valueOf(ret_predict);
        }
        System.out.println("result:value="+strRet);
    	return strRet;
    }
	
	//缩放图片到mnist数据集中图片一致的大小，28x28
	public void scaleToMnist(String srcImageFile,String result) {
		try  
        {  
            BufferedImage src = ImageIO.read(new File(srcImageFile));  
            float width = src.getWidth(); 
            float height = src.getHeight();
           
            float scale = 28 / width;
            width = width * scale;  
            height = height * scale;  
           
            Image image = src.getScaledInstance((int)width, (int)height, Image.SCALE_SMOOTH);  
            BufferedImage tag = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);  
            Graphics g = tag.getGraphics();  
            g.drawImage(image, 0, 0, null); 
            g.dispose();  
            ImageIO.write(tag, "JPEG", new File(result));
        }  
        catch (IOException e)  
        {  
            e.printStackTrace();  
        }  
	}	
	
	//将base64编码的图片数据保存为jpg格式
    public boolean ConverBase64ToImage(String imgStr,String strImageFilePath) {  
        if (imgStr == null) 
            return false;  
        BASE64Decoder decoder = new BASE64Decoder();  
        try   
        {  
            byte[] b = decoder.decodeBuffer(imgStr);  
            for(int i=0;i<b.length;++i)  
            {  
                if(b[i]<0)  
                {
                    b[i]+=256;  
                } 
            }             
                     
            OutputStream out = new FileOutputStream(strImageFilePath);      
            out.write(b);  
            out.flush();  
            out.close();  
            return true;  
        }   
        catch (Exception e)   
        {  
            return false;  
        }  
    }
    
    //图片灰度化处理,并保存到约定的临时文件中
    public void imageGrayScale(String image){  
        int[] rgb = new int[3];  
        File file = new File(image);  
        BufferedImage bi = null;  
        try {  
            bi = ImageIO.read(file);  
        } catch (Exception e) {  
            e.printStackTrace();  
        }  
        int width = bi.getWidth();  
        int height = bi.getHeight(); 
        
        double[] pix = new double[width*height];         
        
        int r_min = 10000;
        int r_max = 0;
        int c_min = 10000;
        int c_max = 0;
        int k = 0;
        for (int i = 0; i < height; i++) {  
            for (int j = 0; j < width; j++) {  
                int pixel = bi.getRGB(j, i);  
                rgb[0] = (pixel & 0xff0000) >> 16;  
                rgb[1] = (pixel & 0xff00) >> 8;  
                rgb[2] = (pixel & 0xff);  
               
                // Gray = (R*30 + G*59 + B*11 + 50) / 100
                int g = (rgb[0]*30+rgb[1]*59+rgb[2]*11+50) / 100;
                
                pix[k] = 0;
                if(rgb[0] >= 60){
                	pix[k] = 0.93;
                	
                	if(i <= r_min) r_min = i;
                	if(i >= r_max) r_max = i;
                	if(j <= c_min) c_min = j;
                	if(j >= c_max) c_max = j;
                }    
                k++;
            }
        }  
        
        //图片中心化，将手写内容移动正中间
        System.out.println(r_min+" "+r_max+" "+c_min+" "+c_max);
        if(r_min < 0) r_min = 0;
        if(r_max > height) r_max = height;
        if(c_min < 0) c_min = 0;
        if(c_max > width) c_max = width;
        int cx = (width-(c_max-c_min)) / 2 - c_min;
        int cy = (height-(r_max-r_min)) / 2 - r_min;         
        double[] ret = new double[width*height+1];
        for(int i = 0;i < width*height;i++) {
        	if(pix[i] != 0) {
        		ret[i+(cy*width+cx)] = pix[i];
        	}         	
        }
        ret[width*height] = 1;
               
	    //
        byte[] bytes = new byte[width*height];
        for(int i = 0;i < width*height;i++) {
        	bytes[i] = 0;
        	if(ret[i] != 0) {
        		bytes[i] = 127;
        	}         	
        }
       
        //保存为约定文件路径
        FileOutputStream fos = null;
        DataOutputStream dos = null;
        try {
        	File dir = new File("/root/temp");
        	if(!dir.exists()) {
        		dir.mkdir();
        	}
        	String strImageData = "/root/temp/imagedata";
			fos = new FileOutputStream(strImageData);
	        dos = new DataOutputStream(fos);
	        for(int i= 0;i < bytes.length;i++) {
	        	dos.writeByte(bytes[i]);
	        }
	        dos.flush();
	        dos.close();  
	        fos.close();
        } catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}        
    }  
}
