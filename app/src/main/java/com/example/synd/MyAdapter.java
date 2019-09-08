package com.example.synd;

import android.content.Context;
import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.util.List;

public class MyAdapter extends RecyclerView.Adapter<MyAdapter.Viewholder> {
    List <ListItem> list;
    Context context;

    public MyAdapter(List <ListItem> list, Context context) {
        this.list = list;
        this.context = context;
    }

    @NonNull
    @Override
    public MyAdapter.Viewholder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.listitem, parent, false);
        return new Viewholder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull MyAdapter.Viewholder holder, int position) {
        ListItem l = list.get(position);
        holder.txttitle.setText(l.getTitle());
        holder.txtdate.setText(l.getDate());
        holder.txtid.setText(l.getId());

    }
    @Override
    public int getItemCount() {
        return list.size();
    }
    public class Viewholder extends RecyclerView.ViewHolder{
        TextView txttitle;
        TextView txtdate ;
        TextView txtid ;

        public Viewholder(View itemView) {
            super(itemView);
            txttitle = (TextView)itemView.findViewById(R.id.txttitle);
            txtdate =(TextView)itemView.findViewById(R.id.txtdate);
            txtid =(TextView)itemView.findViewById(R.id.txtid);
        }
    }
}
