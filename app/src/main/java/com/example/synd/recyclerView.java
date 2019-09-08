package com.example.synd;


import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.widget.Toast;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;
import java.util.List;

public class recyclerView extends AppCompatActivity {
    private RecyclerView recyclerView;
    private RecyclerView.Adapter adapter;
    private List <ListItem> arraylist;
    DatabaseReference mDatabase1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.recyclerview);
        recyclerView =(RecyclerView)findViewById(R.id.recyclerview);
        recyclerView.setHasFixedSize(true);
        recyclerView.setLayoutManager(new LinearLayoutManager(getApplicationContext()));
        arraylist = new ArrayList <>();
        mDatabase1 = FirebaseDatabase.getInstance().getReference().child("+916351511532");

      mDatabase1.addValueEventListener(new ValueEventListener() {
          @Override
          public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
              for(DataSnapshot post:dataSnapshot.getChildren())
              {

                  ListItem listItem=new ListItem();
                   listItem.setTitle(post.child("Complaint").getValue().toString());
                   listItem.setDate(post.child("Respone").getValue().toString());
                   Toast.makeText(recyclerView.this, post.child("Respone").getValue().toString(), Toast.LENGTH_SHORT).show();
                   listItem.setId(post.child("Status").getValue().toString());
            arraylist.add(listItem);

              }
              adapter=new MyAdapter(arraylist,getApplicationContext());
              recyclerView.setAdapter(adapter);
          }

          @Override
          public void onCancelled(@NonNull DatabaseError databaseError) {

          }
      });


    }
}

