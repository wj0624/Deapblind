package com.example.deepblind;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.util.Locale;

import android.speech.tts.TextToSpeech;
import android.speech.tts.TextToSpeech.OnInitListener;

public class MainActivity extends AppCompatActivity implements OnInitListener {

    private static final int PERMISSION_REQUEST_CODE = 1;
    private TextToSpeech tts;
    Uri selectedImageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tts = new TextToSpeech(this, this);

        Button galleryBtn = findViewById(R.id.galleryBtn);
        galleryBtn.setOnClickListener(view -> {
            tts.speak("갤러리에서 이미지를 가져옵니다", TextToSpeech.QUEUE_FLUSH, null, null);
            getImageFromGallery();
        });

        Button cameraBtn = findViewById(R.id.cameraBtn);
        cameraBtn.setOnClickListener(view -> {
            tts.speak("카메라를 실행합니다.", TextToSpeech.QUEUE_FLUSH, null, null);
            getImageFromCamera();
        });
    }
    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            int result = tts.setLanguage(Locale.KOREAN); // 원하는 언어로 설정

            if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                // 언어 데이터가 없거나 지원되지 않을 경우 처리
            } else {
                // TTS를 사용하여 텍스트를 읽기
                tts.speak("안녕하세요, 음성안내 시스템 딥블라인드입니다.", TextToSpeech.QUEUE_FLUSH, null, null);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tts != null) {
            tts.stop();
            tts.shutdown();
            tts = null;
        }
    }

    //갤러리에서 가지고 오기
    private final ActivityResultLauncher<Intent> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {

                if (result.getResultCode() == Activity.RESULT_OK) {
                    Intent data = result.getData();
                    if (data == null) {
                        return;
                    }

                    Uri selectedImage = data.getData();

                    Intent intent = new Intent(this, GalleryActivity.class);
                    intent.setData(selectedImage);
                    startActivity(intent);
                }
            });

    private void getImageFromGallery() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT).setType("image/*");
        galleryLauncher.launch(intent);
    }//갤러리

    //카메라에서 가지고 오기
    private final ActivityResultLauncher<Intent> activityResultPicture = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result ->  {
                if(result.getResultCode()== Activity.RESULT_OK){
                    if (selectedImageUri == null) {
                        return;
                    }

                    Intent intent = new Intent(this, CameraActivity.class);
                    intent.setData(selectedImageUri);
                    startActivity(intent);
                }
            });
    
    private void getImageFromCamera(){
        if (checkCameraPermission()) {
            File file = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "picture.jpg");
            if (file.exists()) {
                file.delete();
            }
            selectedImageUri = FileProvider.getUriForFile(this, getPackageName(), file);
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, selectedImageUri);
            activityResultPicture.launch(intent);
        }
    }//카메라
    private boolean checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            // 이미 카메라 권한이 부여되어 있는 경우
            return true;
        } else {
            // 카메라 권한을 요청
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CODE);
            return false;
        }
    }
}