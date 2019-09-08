package com.example.synd;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import com.example.synd.Admin.Admin_recyclerView;

import java.util.Map;

public class Choose extends AppCompatActivity {
Button btn1,btn2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_choose);
        btn1 =(Button)findViewById(R.id.user);
        btn2=(Button)findViewById(R.id.admin);
        SharedPreferences sharedPreferences = getSharedPreferences("who", Context.MODE_PRIVATE);
        final SharedPreferences.Editor editor = sharedPreferences.edit();

        btn1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                editor.putString("1", "1");

                editor.apply();

                Intent intent1=new Intent(Choose.this, Speech.class);
                startActivity(intent1);

            }
        });
        btn2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                editor.putString("2", "2");

                editor.apply();


                Intent intent1=new Intent(Choose.this, MainActivity.class);
                startActivity(intent1);

            }
        });

    }
}
