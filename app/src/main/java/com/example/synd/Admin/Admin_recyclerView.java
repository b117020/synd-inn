package com.example.synd.Admin;


import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.widget.Toast;

import com.example.synd.R;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;
import java.util.List;

public class Admin_recyclerView extends AppCompatActivity {
    private RecyclerView recyclerView;
    private RecyclerView.Adapter adapter;
    private List <Admin_ListItem> arraylist;
    DatabaseReference mDatabase1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.recyclerview);
        recyclerView =(RecyclerView)findViewById(R.id.recyclerview);
        recyclerView.setHasFixedSize(true);
        recyclerView.setLayoutManager(new LinearLayoutManager(getApplicationContext()));
        arraylist = new ArrayList <>();
        mDatabase1 = FirebaseDatabase.getInstance().getReference();

      mDatabase1.addValueEventListener(new ValueEventListener() {
          @Override
          public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
              for(DataSnapshot post:dataSnapshot.getChildren())
              {
                 for(DataSnapshot post1:post.getChildren())
                 {

                     Admin_ListItem adminListItem =new Admin_ListItem();

                        adminListItem.setPhone(post.getKey().toString());
                        adminListItem.setUID(post1.getKey());
                     adminListItem.setTitle(post1.child("Complaint").getValue().toString());
                     adminListItem.setDate(post1.child("Respone").getValue().toString());
//                     Toast.makeText(Admin_recyclerView.this, post.child("Respone").getValue().toString(), Toast.LENGTH_SHORT).show();
                     adminListItem.setId(post1.child("Status").getValue().toString());
                     arraylist.add(adminListItem);

                 }

              }
              adapter=new Admin_MyAdapter(arraylist,getApplicationContext());
              recyclerView.setAdapter(adapter);
          }

          @Override
          public void onCancelled(@NonNull DatabaseError databaseError) {

          }
      });


    }
}

