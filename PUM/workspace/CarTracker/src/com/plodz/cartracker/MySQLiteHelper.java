package com.plodz.cartracker;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteDatabase.CursorFactory;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

public class MySQLiteHelper extends SQLiteOpenHelper {

	public static final String TABLE_TRIPS = "trips";
	public static final String COLUMN_ID = "_id";
	public static final String COLUMN_NODES = "nodes";
	public static final String COLUMN_NODESLL = "nodesll";
	public static final String COLUMN_STARTTIME = "starttime";
	public static final String COLUMN_ENDTIME = "endtime";
	public static final String COLUMN_STARTADDRESS = "startaddress";
	public static final String COLUMN_ENDADDRESS = "endaddress";
	public static final String COLUMN_AVGSPEED = "avgspeed";
	public static final String COLUMN_DISTANCE = "distance";
	public static final String COLUMN_FUELCONSUMED = "fuelconsumed";
	public static final String COLUMN_FUELCOST = "fuelcost";
	
	public static final String TABLE_FUELS = "fuels";
	public static final String COLUMN_FUEL_ID = "_id";
	public static final String COLUMN_FUEL_TYPE = "fueltype";
	public static final String COLUMN_FUEL_PRICE = "fuelprice";
	
	public static final String TABLE_MISC = "misc";
	public static final String COLUMN_MISC_ID = "_id";
	public static final String COLUMN_MISC_MYFUEL = "myfuel";
	public static final String COLUMN_MISC_MYCONS = "myconsumption";
	public static final String COLUMN_MISC_UPDRATIO = "updratio";
	public static final String COLUMN_MISC_CHECKDELAY = "checkdelay";
	public static final String COLUMN_MISC_LASTUPDATE = "lastupdate";
	public static final String COLUMN_MISC_ZOOMMULTIPLIER = "zoommultiplier";
	
	
	private static final String DATABASE_NAME = "cartracker.db";
	private static final int DATABASE_VERSION = 1;
	
	// Database creation SQL statement
	private static final String DATABASE_CREATE_01 = "create table "
			+ TABLE_TRIPS + "(" + COLUMN_ID
			+ " integer primary key autoincrement, " 
			+ COLUMN_NODES + " text not null, "
			+ COLUMN_NODESLL + " text not null, "
			+ COLUMN_STARTTIME + " integer not null, "
			+ COLUMN_ENDTIME + " integer not null, "
			+ COLUMN_STARTADDRESS + " text not null, "
			+ COLUMN_ENDADDRESS + " text not null, "
			+ COLUMN_AVGSPEED + " real not null, "
			+ COLUMN_DISTANCE + " real not null, "
			+ COLUMN_FUELCONSUMED + " real not null, "
			+ COLUMN_FUELCOST + " real not null); ";
	private static final String DATABASE_CREATE_02 = "create table "
			+ TABLE_FUELS + "(" + COLUMN_FUEL_ID
			+ " integer primary key autoincrement, " 
			+ COLUMN_FUEL_TYPE + " text not null, "
			+ COLUMN_FUEL_PRICE + " real not null); ";
	private static final String DATABASE_CREATE_03 = "create table "
			+ TABLE_MISC + "(" + COLUMN_MISC_ID + " integer primary key autoincrement, "
			+ COLUMN_MISC_MYFUEL + " integer not null, "
			+ COLUMN_MISC_MYCONS + " real not null, "
			+ COLUMN_MISC_UPDRATIO + " real not null, "
			+ COLUMN_MISC_CHECKDELAY + " integer not null, "
			+ COLUMN_MISC_LASTUPDATE + " integer not null, "
			+ COLUMN_MISC_ZOOMMULTIPLIER + " real not null); ";
	
	public MySQLiteHelper(Context context) 
	{
		super(context, DATABASE_NAME, null, DATABASE_VERSION);
	}

	@Override
	public void onCreate(SQLiteDatabase db) 
	{
		db.execSQL(DATABASE_CREATE_01);
		db.execSQL(DATABASE_CREATE_02);
		db.execSQL(DATABASE_CREATE_03);
	}

	@Override
	public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) 
	{
		Log.w(MySQLiteHelper.class.getName(), " Upgrading database from version " + 
				oldVersion + " to " + newVersion + ", which will destroy all old data");
		db.execSQL("drop table if exists " + TABLE_TRIPS);
		db.execSQL("drop table if exists " + TABLE_FUELS);
		onCreate(db);
	}

}
