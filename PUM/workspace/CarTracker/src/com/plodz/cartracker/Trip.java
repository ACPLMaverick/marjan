package com.plodz.cartracker;

import java.io.StringReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Locale;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.os.AsyncTask;
import android.text.format.Formatter;
import android.text.format.Time;

import com.google.android.gms.maps.model.LatLng;

public class Trip {
	
	public ArrayList<Location> nodes;
	public ArrayList<LatLng> nodesLL;
	public LatLng currentPos;
	public float currentSpeed = 0;
	public float avgSpeed = 0;
	public GregorianCalendar startTime;
	public GregorianCalendar currentTime;
	public GregorianCalendar endTime;
	public float distance = 0;
	public float fuelConsumed = 0;
	public float fuelCost = 0;
	public String startAddress;
	public String endAddress;
	
	private Context acContext; 
	
	public Trip()
	{
		nodes = new ArrayList<Location>();
		nodesLL = new ArrayList<LatLng>();
		startTime = new GregorianCalendar();
		startTime.setTimeInMillis(new Date().getTime());
		currentTime = new GregorianCalendar();
		currentTime.set(GregorianCalendar.HOUR_OF_DAY, 0);
		currentTime.set(GregorianCalendar.MINUTE, 0);
		currentTime.set(GregorianCalendar.SECOND, 0);
		endTime = new GregorianCalendar();
		startAddress = "???";
		endAddress = "???";
	}
	
	public Trip(Context acContext)
	{
		this();
		this.acContext = acContext;
	}
	
	public Trip(TripModel tm)
	{
		nodes = new ArrayList<Location>();
		nodesLL = new ArrayList<LatLng>();
		JSONObject jsonNodes = null;
		JSONObject jsonLL = null;
		try
		{
			jsonNodes = new JSONObject(tm.getNodes());
			jsonLL = new JSONObject(tm.getNodesLL());
		}
		catch(JSONException e)
		{
			e.printStackTrace();
		}
		
		if(jsonNodes != null)
		{
			 JSONArray list = jsonNodes.optJSONArray("nodes");
			 int l = list.length();
			 for(int i = 0; i < l; i++)
			 {
				 try
				 {
					 nodes.add(getLocationFromString(list.getString(i)));
				 }
				 catch(JSONException e)
				{
					e.printStackTrace();
				}
			 }
		}
		
		if(jsonLL != null)
		{
			 JSONArray list = jsonLL.optJSONArray("nodesLL");
			 int l = list.length();
			 for(int i = 0; i < l; i++)
			 {
				 try
				 {
					 nodesLL.add(getLatLngFromString(list.getString(i)));
				 }
				 catch(JSONException e)
				{
					e.printStackTrace();
				}
			 }
		}
		
		startTime = new GregorianCalendar();
		startTime.setTimeInMillis(tm.getStartTime());
		currentTime = new GregorianCalendar();
		endTime = new GregorianCalendar();
		endTime.setTimeInMillis(tm.getEndTime());
		startAddress = tm.getStartAddress();
		endAddress = tm.getEndAddress();
		currentSpeed = 0.0f;
		avgSpeed = (float) tm.getAvgSpeed();
		distance = (float) tm.getDistance();
		fuelConsumed = (float) tm.getFuelConsumed();
		fuelCost = (float) tm.getFuelCost();
	}
	
	public void update(Location loc, long millis)
	{
		nodes.add(loc);
		updateTimeOnly(millis);
		
		currentSpeed = loc.getSpeed();
		nodesLL.add(new LatLng(loc.getLatitude(), loc.getLongitude()));
		currentPos = nodesLL.get(nodesLL.size() - 1);
		if(nodes.size() > 1)
		{
			currentSpeed = loc.getSpeed()*3.6f;
			updateDistance();
			updateFuelConsumed();
			updateFuelCost();
			updateAvgSpeed();
		}
		else
		{
			startAddress = getAdddessFromLocation(nodes.get(0));
		}
	}
	
	public void end()
	{
		if(nodes.size() > 0) endAddress = getAdddessFromLocation(nodes.get(nodes.size() - 1));
		else endAddress = startAddress;
		endTime.setTime(new Date());
	}
	
	public void updateTimeOnly(long millis)
	{
//		currentTime.setTimeInMillis((new Date()).getTime() - startTime.getTimeInMillis() - 3600000);
		currentTime.setTimeInMillis(currentTime.getTimeInMillis() + millis);
//		endTime.setTime(new Date());
	}
	
	public String[] getStatsAsStringArray()
	{
		String[] stats = new String[TrackingController.STAT_COUNT];
		stats[0] = String.format("%.2f", fuelConsumed);
		stats[1] = String.format("%.2f", fuelCost);
		stats[2] = String.format("%.2f", distance);
		stats[3] = String.format("%.2f", avgSpeed);
		stats[4] = String.format("%.2f", currentSpeed);
		stats[5] = String.format("%tT", startTime);
		stats[6] = String.format("%tT", currentTime);
		
		return stats;
	}

	private void updateAvgSpeed()
	{
//		float average = 0;
//		for(Location loc : nodes)
//		{
//			average += loc.getSpeed();
//		}
//		average /= nodes.size();
//		avgSpeed = average;
		
		float t = currentTime.get(GregorianCalendar.SECOND) + 60*currentTime.get(GregorianCalendar.MINUTE) + 
				3600*currentTime.get(GregorianCalendar.HOUR);
		t = t/3600.0f;
		avgSpeed = (distance / t);
	}

	private void updateFuelCost()
	{
		float currentPrice;
		if(Globals.myFuelType == Globals.fuelType.LPG) currentPrice = Globals.priceLPG;
		else if(Globals.myFuelType == Globals.fuelType.ON) currentPrice = Globals.priceON;
		else if(Globals.myFuelType == Globals.fuelType.PB98) currentPrice = Globals.pricePB98;
		else if(Globals.myFuelType == Globals.fuelType.PB95) currentPrice = Globals.pricePB95;
		else currentPrice = 0.0f;
		
		fuelCost = currentPrice*fuelConsumed;
	}
	
	private void updateFuelConsumed()
	{
		float consumption = Globals.myFuelConsumption;
		fuelConsumed = (distance*consumption)/100.0f;
	}
	
	private void updateDistance()
	{
		float earthRadius = 6378.137f;
		
		int i = nodes.size() - 1;
			double lat2 = nodes.get(i).getLatitude();
			double lon2 = nodes.get(i).getLongitude();
			double lat1 = nodes.get(i - 1).getLatitude();
			double lon1 = nodes.get(i - 1).getLongitude();
			
			double dLat = (lat2 - lat1) * Math.PI / 180;
			double dLon = (lon2 - lon1) * Math.PI / 180;
			
			// EEEEEEEEEEEEEEEEEE MACARENA!!!
			double a = Math.sin(dLat/2) * Math.sin(dLat/2) +
				    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
				    Math.sin(dLon/2) * Math.sin(dLon/2);
			double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
			double d = earthRadius * c;
			
		distance += d;
	}
	
	private String getAdddessFromLocation(Location loc)
	{
		if(acContext == null) return "";
		
		GetAddressTask task = new GetAddressTask(acContext);
		return task.doInBackground(loc);
	}
	
	@SuppressLint("SimpleDateFormat") @Override
	public String toString()
	{
		SimpleDateFormat sdf = new SimpleDateFormat("dd.MM.yy HH:mm");
		String startDate = sdf.format(startTime.getTime());
		String endDate = sdf.format(endTime.getTime());
		
		String toReturn = startDate + " - " + endDate + "\n"
				+ startAddress + " - " + endAddress;
		
		return toReturn;
	}
	
	private Location getLocationFromString(String str)
	{
		String[] strings = str.split(";");
		
		double lat = Double.valueOf(strings[0]);
		double lon = Double.valueOf(strings[1]);
		float spd = Float.valueOf(strings[2]);
		double alt = Double.valueOf(strings[3]);
		float brg = Float.valueOf(strings[4]);
		long time = Long.valueOf(strings[5]);
		String provider = strings[6];
		
		Location loc = new Location(provider);
		loc.setLatitude(lat);
		loc.setLongitude(lon);
		loc.setSpeed(spd);
		loc.setAltitude(alt);
		loc.setBearing(brg);
		loc.setTime(time);
		
		return loc;
	}
	
	private LatLng getLatLngFromString(String str)
	{
		String[] strings = str.split(";");
		
		double lat = Double.valueOf(strings[0]);
		double lon = Double.valueOf(strings[1]);
		
		return new LatLng(lat, lon);
	}
}
