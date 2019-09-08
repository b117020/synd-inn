package com.example.synd;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.Patterns;
import android.view.View;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;

public class s_login extends AppCompatActivity implements View.OnClickListener {
    EditText userm;            //MainActivity=LoginPage
    EditText passm;
    FirebaseAuth mAuth;
    String ema;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);
        mAuth=FirebaseAuth.getInstance();
        userm =(EditText) findViewById(R.id.userm);
        passm =(EditText)findViewById(R.id.passm);
        SharedPreferences sharedPreferences2 = getSharedPreferences("email", Context.MODE_PRIVATE);
//        final String  email = sharedPreferences2.getString("email", "");
//        if(e) {
//            Intent intent=new Intent(s_login.this, Speech.class );
//            startActivity(intent);
//        }

        findViewById(R.id.signupbtn).setOnClickListener(this);
        findViewById(R.id.loginbtn).setOnClickListener(this);
    }
    private void userlogin() {
        ema= userm.getText().toString().trim();
        String pas= passm.getText().toString().trim();
        if(ema.isEmpty())
        {
            userm.setError("Email is required...");
            userm.requestFocus();
            return;   //returning to calling funtion
        }
        if(pas.isEmpty())
        {
            passm.setError("Password is required...");
            passm.requestFocus();
            return;   //returing to calling functoin
        }
        if(!Patterns.EMAIL_ADDRESS.matcher(ema).matches())
        {
            userm.setError("Enter a vadit email address");
            userm.requestFocus();
            return;
        }
        if(passm.length()<6)
        {
            passm.setError("Mini. required password length is 6");
            passm.requestFocus();
            return;
        }


        mAuth.signInWithEmailAndPassword(ema,pas).addOnCompleteListener(new OnCompleteListener <AuthResult>() {
            @Override
            public void onComplete(@NonNull Task <AuthResult> task) {

                if(task.isSuccessful())
                {
                    SharedPreferences sharedPreferences = getSharedPreferences("email", Context.MODE_PRIVATE);
                    SharedPreferences.Editor editor = sharedPreferences.edit();
                    editor.putString("email", ema);
                    editor.apply();
                    Toast.makeText(s_login.this,"onComplete",Toast.LENGTH_LONG).show();

                    Intent i=new Intent(s_login.this,Speech.class);
                    startActivity(i);
                }
                else{
                    Toast.makeText(s_login.this,"Something went wrong", Toast.LENGTH_SHORT).show();
                }}
        });
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.signupbtn:
                startActivity(new Intent(s_login.this,s_signup.class));
                break;
            case R.id.loginbtn:

                userlogin();
                break;
        }
    }
}