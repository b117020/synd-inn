package com.example.synd;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import com.example.synd.Admin.Admin_recyclerView;

public class MainActivity extends AppCompatActivity {
    EditText d;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
      d =(EditText)findViewById(R.id.Admincode);

        Button b=(Button)findViewById(R.id.Adminsub);

        b.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(d.getText().toString().equals("121"))
                {
                    Intent intent1=new Intent(MainActivity.this, Admin_recyclerView.class);
                    startActivity(intent1);

                }

            }
        });


    }
}
