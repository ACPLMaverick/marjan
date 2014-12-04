package com.plodz.cartracker;

import java.util.ArrayList;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;

public class DataSource {
	private SQLiteDatabase database;
	private MySQLiteHelper dbHelper;
	private String[] allColumnsTrip = {
		MySQLiteHelper.COLUMN_ID,
		MySQLiteHelper.COLUMN_NODES,
		MySQLiteHelper.COLUMN_NODESLL,
		MySQLiteHelper.COLUMN_STARTTIME,
		MySQLiteHelper.COLUMN_ENDTIME,
		MySQLiteHelper.COLUMN_STARTADDRESS,
		MySQLiteHelper.COLUMN_ENDADDRESS,
		MySQLiteHelper.COLUMN_AVGSPEED,
		MySQLiteHelper.COLUMN_DISTANCE,
		MySQLiteHelper.COLUMN_FUELCONSUMED,
		MySQLiteHelper.COLUMN_FUELCOST	
	};
	private String[] allColumnsFuel = {
		MySQLiteHelper.COLUMN_FUEL_ID,
		MySQLiteHelper.COLUMN_FUEL_TYPE,
		MySQLiteHelper.COLUMN_FUEL_PRICE
	};
	
	public DataSource(Context context)
	{
		dbHelper = new MySQLiteHelper(context);
	}
	
	public void open() throws SQLException
	{
		database = dbHelper.getWritableDatabase();
	}
	
	public void close()
	{
		dbHelper.close();
	}
	
	public long addTripModel(TripModel tm)
	{
		ContentValues values = new ContentValues();
		values.put(MySQLiteHelper.COLUMN_NODES, tm.getNodes());
		values.put(MySQLiteHelper.COLUMN_NODESLL, tm.getNodesLL());
		values.put(MySQLiteHelper.COLUMN_STARTTIME, tm.getStartTime());
		values.put(MySQLiteHelper.COLUMN_ENDTIME, tm.getEndTime());
		values.put(MySQLiteHelper.COLUMN_STARTADDRESS, tm.getStartAddress());
		values.put(MySQLiteHelper.COLUMN_ENDADDRESS, tm.getEndAddress());
		values.put(MySQLiteHelper.COLUMN_AVGSPEED, tm.getAvgSpeed());
		values.put(MySQLiteHelper.COLUMN_DISTANCE, tm.getDistance());
		values.put(MySQLiteHelper.COLUMN_FUELCONSUMED, tm.getFuelConsumed());
		values.put(MySQLiteHelper.COLUMN_FUELCOST, tm.getFuelCost());
		
		return database.insert(MySQLiteHelper.TABLE_TRIPS, null, values);
	}
	
	public ArrayList<TripModel> getAllTripModels()
	{
		ArrayList<TripModel> list = new ArrayList<TripModel>();
		
		Cursor cursor = database.query(MySQLiteHelper.TABLE_TRIPS, allColumnsTrip, null, null, null, null, null);
		cursor.moveToFirst();
		while(!cursor.isAfterLast())
		{
			list.add(cursorToTripModel(cursor));
			cursor.moveToNext();
		}
		
		cursor.close();
		return list;
	}
	
	public void clearTripTable()
	{
		database.delete(MySQLiteHelper.TABLE_TRIPS, null, null);
	}
	
	public void clearFuelTable()
	{
		database.delete(MySQLiteHelper.TABLE_FUELS, null, null);
	}
	
	private TripModel cursorToTripModel(Cursor cursor)
	{
		return new TripModel(cursor.getLong(0),
							cursor.getString(1), 
							cursor.getString(2),
							cursor.getLong(3),
							cursor.getLong(4),
							cursor.getString(5),
							cursor.getString(6),
							cursor.getDouble(7),
							cursor.getDouble(8),
							cursor.getDouble(9),
							cursor.getDouble(10));
	}
}
