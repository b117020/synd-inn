package com.example.synd;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Locale;

public class Speech extends AppCompatActivity {
  TextView txt, submit;
    DatabaseReference mDatabase;
EditText editText;
Button btn;
     SharedPreferences sharedPreferences2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_speech);
        btn =findViewById(R.id.logout);
        txt =(TextView) findViewById(R.id.Dash);
         submit =(TextView) findViewById(R.id.submit);
         sharedPreferences2 = getSharedPreferences("PHONE", Context.MODE_PRIVATE);
        final String  email = sharedPreferences2.getString("email", "");
        final  String pho=sharedPreferences2.getString("PHONE1","");

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent =new Intent(Speech.this,s_signup.class);
                startActivity(intent);
            }
        });


        Toast.makeText(this, email, Toast.LENGTH_SHORT).show();
        if(email.equals("")) {
            Intent intent12=new Intent(Speech.this, s_signup.class );
            startActivity(intent12);
        }


        editText=(EditText) findViewById(R.id.voice);
         txt.setOnClickListener(new View.OnClickListener() {
             @Override
             public void onClick(View v) {
                 Intent i=new Intent(Speech.this,recyclerView.class);
                 startActivity(i);
             }
         });

         submit.setOnClickListener(new View.OnClickListener() {
             @Override
             public void onClick(View v) {
                 mDatabase = FirebaseDatabase.getInstance().getReference().child(pho).push();

                 String url="http://devddm.pythonanywhere.com/synd/"+ editText.getText().toString();
                 JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.GET,url,null ,
                         new Response.Listener<JSONObject>() {
                             @Override
                             public void onResponse(JSONObject response) {
                                 try {
                                     System.out.println(response);
                                     String str = response.getString("data");
                                     Toast.makeText(Speech.this, str, Toast.LENGTH_SHORT).show();
                                     mDatabase.child("Complaint").setValue(editText.getText().toString().trim());
                                     mDatabase.child("Respone").setValue(str);
                                     mDatabase.child("Status").setValue("Inprogress");


                                 } catch (JSONException e) {
                                     e.printStackTrace();
                                 }

                             }
                         }, new Response.ErrorListener() {
                     @Override
                     public void onErrorResponse(VolleyError error) {
                         System.out.println(error);

                     }
                 });

                 Mysingleton.getInstance(getApplicationContext()).addToRequestqueue(jsonObjectRequest);



             }
         });

         checkPermission();


        final SpeechRecognizer mSpeechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);


        final Intent mSpeechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        mSpeechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        mSpeechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE,
                Locale.getDefault());


        mSpeechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle bundle) {

            }

            @Override
            public void onBeginningOfSpeech() {

            }

            @Override
            public void onRmsChanged(float v) {

            }

            @Override
            public void onBufferReceived(byte[] bytes) {

            }

            @Override
            public void onEndOfSpeech() {

            }

            @Override
            public void onError(int i) {

            }

            @Override
            public void onResults(Bundle bundle) {
                //getting all the matches
                ArrayList <String> matches = bundle
                        .getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);

                //displaying the first match
                if (matches != null)
                    editText.setText(matches.get(0));
            }

            @Override
            public void onPartialResults(Bundle bundle) {

            }

            @Override
            public void onEvent(int i, Bundle bundle) {

            }
        });

        findViewById(R.id.button).setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                switch (motionEvent.getAction()) {
                    case MotionEvent.ACTION_UP:
                        mSpeechRecognizer.stopListening();
                        editText.setHint("You will see input here");
                        break;

                    case MotionEvent.ACTION_DOWN:
                        mSpeechRecognizer.startListening(mSpeechRecognizerIntent);
                        editText.setText("");
                        editText.setHint("Listening...");
                        break;
                }
                return false;
            }
        });

    }

    private void checkPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!(ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)) {
                Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.parse("package:" + getPackageName()));
                startActivity(intent);
                finish();
            }
        }
    }
}
