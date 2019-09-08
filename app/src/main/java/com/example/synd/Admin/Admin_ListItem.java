package com.example.synd.Admin;

public class Admin_ListItem {
    private String title;
    private String date;
    private String id;


    private  String UID;


    private String phone;
 public Admin_ListItem()
 {

 }
    public Admin_ListItem(String title, String date, String id, String phone, String UID) {
        this.title = title;
        this.date = date;
        this.id = id;
        this.phone=phone;
        this.UID=UID;
    }

    public String getUID() {
        return UID;
    }

    public void setUID(String UID) {
        this.UID = UID;
    }


    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }
}
