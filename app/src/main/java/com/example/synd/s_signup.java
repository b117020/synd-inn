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
import android.widget.Toast;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseAuthEmailException;

public class s_signup extends AppCompatActivity implements View.OnClickListener {
    EditText email1;
    EditText pass1;
    FirebaseAuth mAuth;
    String ema;
EditText phone;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_s_signup);
        mAuth=FirebaseAuth.getInstance();
        phone=(EditText)findViewById(R.id.phone);
        email1=(EditText)findViewById(R.id.email1);
        pass1=(EditText)findViewById(R.id.pass1);





//        SharedPreferences sharedPreferences2 = getSharedPreferences("email", Context.MODE_PRIVATE);
//        final String  email = sharedPreferences2.getString("email", "");
//        if(email!="") {
//            Intent intent=new Intent(s_signup.this, Speech.class );
//            startActivity(intent);
//        }


        findViewById(R.id.signup1).setOnClickListener(this);
        findViewById(R.id.login1).setOnClickListener(this);

    }

    private void registeruser() {
        ema=email1.getText().toString().trim();
        String pas=pass1.getText().toString().trim();
        if(ema.isEmpty())
        {
            email1.setError("Email is required...");
            email1.requestFocus();
            return;   //returning to calling funtion
        }
        if(pas.isEmpty())
        {            pass1.setError("Password is required...");
            pass1.requestFocus();
            return;   //returing to calling functoin
        }
        if(!Patterns.EMAIL_ADDRESS.matcher(ema).matches())
        {
            email1.setError("Enter a vadit email address");
            email1.requestFocus();
            return;
        }
        if(pass1.length()<6)
        {
            pass1.setError("Mini. required password length is 6");
            pass1.requestFocus();
            return;
        }
        mAuth.createUserWithEmailAndPassword(ema,pas).addOnCompleteListener(new OnCompleteListener <AuthResult>() {
            @Override
            public void onComplete(@NonNull Task <AuthResult> task) {
                if(task.isSuccessful())
                {
                    SharedPreferences sharedPreferences = getSharedPreferences("email", Context.MODE_PRIVATE);
                    SharedPreferences.Editor editor = sharedPreferences.edit();
                    editor.putString("email", ema);
                    editor.putString("phone", phone.getText().toString().trim());

                    editor.apply();
                    Toast.makeText(s_signup.this, "User register successfull", Toast.LENGTH_SHORT).show();
                    Intent intent =new Intent(s_signup.this,s_login.class);
                    startActivity(intent);
                }
                else{
                    if(task.getException() instanceof FirebaseAuthEmailException){
                        Toast.makeText(s_signup.this, "You are already register1", Toast.LENGTH_SHORT).show();
                    }
                    else{
                        Toast.makeText(getApplicationContext(),"You are already register2",Toast.LENGTH_LONG).show();
                    }
                }
            }
        });

    }


    @Override
    public void onClick(View v) {
        switch (v.getId())
        {
            case R.id.signup1:
                registeruser();
                break;
            case R.id.login1:
                startActivity(new Intent(this,s_login.class));
                break;

        }
    }
}