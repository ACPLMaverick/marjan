package com.plodz.cartracker;

import java.util.ArrayList;
import java.util.Date;
import java.util.GregorianCalendar;

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

	private String[] allColumnsMisc = {
		MySQLiteHelper.COLUMN_MISC_ID,
		MySQLiteHelper.COLUMN_MISC_MYFUEL,
		MySQLiteHelper.COLUMN_MISC_MYCONS,
		MySQLiteHelper.COLUMN_MISC_UPDRATIO,
		MySQLiteHelper.COLUMN_MISC_CHECKDELAY,
		MySQLiteHelper.COLUMN_MISC_LASTUPDATE,
		MySQLiteHelper.COLUMN_MISC_ZOOMMULTIPLIER
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
	
	public void saveFuels()
	{
		clearFuelTable();
		
		ContentValues values = new ContentValues();
		values.put(MySQLiteHelper.COLUMN_FUEL_TYPE, Globals.stringON);
		values.put(MySQLiteHelper.COLUMN_FUEL_PRICE, Globals.priceON);
		database.insert(MySQLiteHelper.TABLE_FUELS, null, values);
		
		values.clear();
		values.put(MySQLiteHelper.COLUMN_FUEL_TYPE, Globals.stringLPG);
		values.put(MySQLiteHelper.COLUMN_FUEL_PRICE, Globals.priceLPG);
		database.insert(MySQLiteHelper.TABLE_FUELS, null, values);
		
		values.clear();
		values.put(MySQLiteHelper.COLUMN_FUEL_TYPE, Globals.stringPB95);
		values.put(MySQLiteHelper.COLUMN_FUEL_PRICE, Globals.pricePB95);
		database.insert(MySQLiteHelper.TABLE_FUELS, null, values);
		
		values.clear();
		values.put(MySQLiteHelper.COLUMN_FUEL_TYPE, Globals.stringPB98);
		values.put(MySQLiteHelper.COLUMN_FUEL_PRICE, Globals.pricePB98);
		database.insert(MySQLiteHelper.TABLE_FUELS, null, values);
	}
	
	public void saveAllGlobals()
	{
		saveFuels();
		clearMiscTable();
		
		ContentValues values = new ContentValues();
		values.put(MySQLiteHelper.COLUMN_MISC_MYFUEL, Globals.myFuelType.ordinal());
		values.put(MySQLiteHelper.COLUMN_MISC_MYCONS, Globals.myFuelConsumption);
		values.put(MySQLiteHelper.COLUMN_MISC_UPDRATIO, Globals.DBG_updateRatio);
		values.put(MySQLiteHelper.COLUMN_MISC_CHECKDELAY, Globals.checkDelay);
		values.put(MySQLiteHelper.COLUMN_MISC_LASTUPDATE, Globals.lastUpdate.getTimeInMillis());
		values.put(MySQLiteHelper.COLUMN_MISC_ZOOMMULTIPLIER, Globals.mapZoomMultiplier);
		database.insert(MySQLiteHelper.TABLE_MISC, null, values);
	}
	
	public boolean loadFuelPrices()
	{
		Cursor cursor = database.query(MySQLiteHelper.TABLE_FUELS, allColumnsFuel, null, null, null, null, null);
		if(cursor.getCount() == 0) return false;
		cursor.moveToFirst();
		
		while(!cursor.isAfterLast())
		{
			String type = cursor.getString(1);
			if(type.equals(Globals.stringON))
			{
				Globals.priceON = cursor.getFloat(2);
			}
			else if(type.equals(Globals.stringLPG))
			{
				Globals.priceLPG = cursor.getFloat(2);
			}
			else if(type.equals(Globals.stringPB95))
			{
				Globals.pricePB95 = cursor.getFloat(2);
			}
			else if(type.equals(Globals.stringPB98))
			{
				Globals.pricePB98 = cursor.getFloat(2);
			}
			cursor.moveToNext();
		}
		
		return true;
	}
	
	public boolean loadMiscGlobals()
	{
		Cursor cursor = database.query(MySQLiteHelper.TABLE_MISC, allColumnsMisc, null, null, null, null, null);
		if(cursor.getCount() == 0) return false;
		cursor.moveToFirst();
		
		Globals.myFuelType = Globals.fuelType.values()[cursor.getInt(1)];
		Globals.myFuelConsumption = cursor.getFloat(2);
		Globals.DBG_updateRatio = cursor.getFloat(3);
		Globals.checkDelay = cursor.getInt(4);
		GregorianCalendar cal = new GregorianCalendar();
		cal.setTime(new Date(cursor.getLong(5)));
		Globals.lastUpdate = cal;
		Globals.mapZoomMultiplier = cursor.getFloat(6);
		
		return true;
	}
	
	public void clearMiscTable()
	{
		database.delete(MySQLiteHelper.TABLE_MISC, null, null);
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
