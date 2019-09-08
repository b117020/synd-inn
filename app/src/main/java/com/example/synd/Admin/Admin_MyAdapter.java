package com.example.synd.Admin;

import android.content.Context;
import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import com.example.synd.R;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.List;

public class Admin_MyAdapter extends RecyclerView.Adapter<Admin_MyAdapter.Viewholder> {
    List <Admin_ListItem> list;
    Context context;
    private int index;
    DatabaseReference mdata;

    public Admin_MyAdapter(List <Admin_ListItem> list, Context context) {
        this.list = list;
        this.context = context;
    }

    @NonNull
    @Override
    public Admin_MyAdapter.Viewholder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.admin_list, parent, false);
        return new Viewholder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull final Admin_MyAdapter.Viewholder holder, final int position) {
        Admin_ListItem l = list.get(position);
        holder.txttitle.setText(l.getTitle());
        holder.txtdate.setText(l.getDate());
        holder.txtid.setText(l.getId());

        index = position;

        holder.btn.setOnClickListener(new View.OnClickListener() {

                                                      @Override
                                                      public void onClick(View view) {
                                                          index = holder.getAdapterPosition();
                                                          mdata = FirebaseDatabase.getInstance().getReference().child(list.get(position).getPhone()).child(list.get(position).getUID());
                                                          mdata.child("Status").setValue("Completed");
                                                      }
                                                  }
        );




    }
    @Override
    public int getItemCount() {
        return list.size();
    }
    public class Viewholder extends RecyclerView.ViewHolder{
        TextView txttitle;
        TextView txtdate ;
        TextView txtid ;
        Button btn;
        public Viewholder(View itemView) {
            super(itemView);
            txttitle = (TextView)itemView.findViewById(R.id.txttitle);
            txtdate =(TextView)itemView.findViewById(R.id.txtdate);
            txtid =(TextView)itemView.findViewById(R.id.txtid);
            btn=(Button)itemView.findViewById(R.id.comp);
        }
    }
}
