package com.example.deepblind;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Log;

import android.widget.ImageView;
import android.widget.TextView;
import android.speech.tts.TextToSpeech;
import android.speech.tts.TextToSpeech.OnInitListener;
import org.json.JSONObject;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.DataOutputStream;
import java.util.Locale;
import java.net.HttpURLConnection;
import java.net.URL;

public class GalleryActivity extends AppCompatActivity implements OnInitListener {
    private static final String TAG = "Read Image";

    private ImageView imageView;
    private TextView textView;
    private TextToSpeech tts;

    //액티비티 진입 시
    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            int result = tts.setLanguage(Locale.KOREAN); // 원하는 언어로 설정

            if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                // 언어 데이터가 없거나 지원되지 않을 경우 처리
            } else {
                tts.speak("글자를 인식 중입니다. 잠시만 기다려주세요", TextToSpeech.QUEUE_FLUSH, null, null);
            }
        }
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        //TTs 객체 중지 및 삭제
        if(tts!=null){
            tts.stop();
            tts.shutdown();
            tts = null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery);

        imageView = findViewById(R.id.selectedImg);
        textView = findViewById(R.id.gal_txt);
        tts = new TextToSpeech(this, this);
        Intent intent = getIntent();
        Uri selectedImage = intent.getData();

        //갤러리에서 가지고 온 이미지 표시
        Bitmap bitmap = null;
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                ImageDecoder.Source src = ImageDecoder.createSource(getContentResolver(), selectedImage);
                bitmap = ImageDecoder.decodeBitmap(src);
            } else {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImage);
            }
        } catch (IOException ioe) {
            Log.e(TAG, "Failed to read Image", ioe);
        }
        imageView.setImageBitmap(bitmap);
        String imgPath = getRealPathFromURI(selectedImage);

        String waitplz = "글자를 인식 중입니다. 잠시만 기다려주세요";
        textView.setText(waitplz);
        tts.speak(waitplz, TextToSpeech.QUEUE_FLUSH, null, null);

        //갤러리에서 가지고 온 이미지 서버에 업로드
        new ImageUploaderTask().execute(imgPath);
    }

    //Main Activity에서 넘겨받은 URI의 저장 경로 가져오기
    private String getRealPathFromURI(Uri uri) {
        String fileName = "my_image.jpg"; // 저장할 파일 이름
        File file = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), fileName);

        try {
            InputStream inputStream = getContentResolver().openInputStream(uri);
            OutputStream outputStream = new FileOutputStream(file);
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }
            outputStream.close();
            inputStream.close();

            // 이제 'file'은 저장된 이미지 파일, file.getAbsolutePath()로 파일의 절대 경로를 얻을 수 있음
            return file.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    //서버에 이미지 업로드하기
    private class ImageUploaderTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            String imagePath = params[0];
            String serverUrl = "http://172.16.194.103:5000/uploader";
            String result = "글자 인식에 실패했습니다. 다른 사진으로 시도해주세요.";

            try {
                File imageFile = new File(imagePath);
                FileInputStream fileInputStream = new FileInputStream(imageFile);
                URL url = new URL(serverUrl);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();

                // HTTP POST 메서드 설정
                connection.setRequestMethod("POST");
                connection.setDoOutput(true);
                // 파일 전송을 위한 경계 문자열 설정
                String boundary = "*****"; // 임의의 경계 문자열
                String lineEnd = "\r\n";

                // HTTP 요청 헤더 설정
                connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
                // 출력 스트림 설정
                DataOutputStream outputStream = new DataOutputStream(connection.getOutputStream());

                // 'file' 키로 이미지 파일 전송 시작
                outputStream.writeBytes("--" + boundary + lineEnd);
                outputStream.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"" + imageFile.getName() + "\"" + lineEnd);
                outputStream.writeBytes("Content-Type: image/jpeg" + lineEnd);
                outputStream.writeBytes(lineEnd);

                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = fileInputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                outputStream.writeBytes(lineEnd);

                // 이미지 파일 전송 종료
                outputStream.writeBytes("--" + boundary + "--" + lineEnd);
                outputStream.flush();
                outputStream.close();
                fileInputStream.close();

                int responseCode = connection.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    InputStream responseStream = connection.getInputStream();
                    BufferedReader reader = new BufferedReader(new InputStreamReader(responseStream));
                    StringBuilder response = new StringBuilder();
                    String line;

                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    reader.close();

                    JSONObject jsonResponse = new JSONObject(response.toString());
                    result = jsonResponse.getString("result");
                }
//                else {
//                    result = "글자 인식에 실패했습니다.";
//                }
                connection.disconnect();
            } catch (Exception e) {
                e.printStackTrace();
            }
            return result;
        }

        @Override
        protected void onPostExecute(String result) {
            textView.setText(result);
            tts.setSpeechRate(0.75f);
            tts.speak(result, TextToSpeech.QUEUE_FLUSH, null, null);
        }
    }
}