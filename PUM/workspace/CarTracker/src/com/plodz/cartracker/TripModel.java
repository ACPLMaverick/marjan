package com.plodz.cartracker;

import java.text.SimpleDateFormat;
import java.util.ArrayList;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.google.android.gms.maps.model.LatLng;

import android.annotation.SuppressLint;
import android.location.Location;

public class TripModel {
	
	private long _id;
	private String nodes;
	private String nodesLL;
	private long startTime;
	private long endTime;
	private String startAddress;
	private String endAddress;
	private double avgSpeed;
	private double distance;
	private double fuelConsumed;
	private double fuelCost;
	
	
	public TripModel(Trip trip)
	{
		JSONObject jsonNodes = new JSONObject();
		JSONObject jsonLL = new JSONObject();
		try {
			jsonNodes.put("nodes", new JSONArray(getLocationsAsString(trip.nodes)));
			jsonLL.put("nodesLL", new JSONArray(getLatLngsAsString(trip.nodesLL)));
		} catch (JSONException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		nodes = jsonNodes.toString();
		nodesLL = jsonLL.toString();
		
		this._id = -1;
		startTime = trip.startTime.getTimeInMillis();
		endTime = trip.endTime.getTimeInMillis();
		startAddress = trip.startAddress;
		endAddress = trip.endAddress;
		avgSpeed = trip.avgSpeed;
		distance = trip.distance;
		fuelConsumed = trip.fuelConsumed;
		fuelCost = trip.fuelCost;
	}
	
	public TripModel(
			long id,
			String nodes, 
			String nodesLL,
			long startTime,
			long endTime,
			String startAddress,
			String endAddress,
			double avgSpeed,
			double distance,
			double fuelConsumed,
			double fuelCost)
	{
		this._id = id;
		this.nodes = nodes;
		this.nodesLL = nodesLL;
		this.startTime = startTime;
		this.endTime = endTime;
		this.startAddress = startAddress;
		this.endAddress = endAddress;
		this.avgSpeed = avgSpeed;
		this.distance = distance;
		this.fuelConsumed = fuelConsumed;
		this.fuelCost = fuelCost;
	}
	
	public long getID() { return _id; };
	public String getNodes() { return nodes; };
	public String getNodesLL() { return nodesLL; };
	public long getStartTime() { return startTime; };
	public long getEndTime() { return endTime; };
	public String getStartAddress() { return startAddress; };
	public String getEndAddress() { return endAddress; };
	public double getAvgSpeed() { return avgSpeed; };
	public double getDistance() { return distance; };
	public double getFuelConsumed() { return fuelConsumed; };
	public double getFuelCost() { return fuelCost; };
	
	private ArrayList<String> getLocationsAsString(ArrayList<Location> list)
	{
		ArrayList<String> ret = new ArrayList<String>();
		for(Location loc: list)
		{
			String toAdd = 
					String.valueOf(loc.getLatitude()) + ";"
					+ String.valueOf(loc.getLongitude()) + ";"
					+ String.valueOf(loc.getSpeed()) + ";"
					+ String.valueOf(loc.getAltitude()) + ";"
					+ String.valueOf(loc.getBearing()) + ";"
					+ String.valueOf(loc.getTime()) + ";"
					+ loc.getProvider();
			ret.add(toAdd);
		}
		return ret;
	}
	
	private ArrayList<String> getLatLngsAsString(ArrayList<LatLng> list)
	{
		ArrayList<String> ret = new ArrayList<String>();
		for(LatLng loc: list)
		{
			String toAdd = 
					String.valueOf(loc.latitude) + ";"
					+ String.valueOf(loc.longitude);
			ret.add(toAdd);
		}
		return ret;
	}
	
	@SuppressLint("SimpleDateFormat") @Override
	public String toString()
	{
		SimpleDateFormat sdf = new SimpleDateFormat("dd.MM.yy HH:mm");
		String startDate = sdf.format(startTime);
		String endDate = sdf.format(endTime);
		
		String toReturn = startDate + " - " + endDate + "\n"
				+ startAddress + " - " + endAddress;
		
		return toReturn;
	}
}
