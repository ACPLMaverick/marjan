package com.plodz.cartracker;

import java.text.SimpleDateFormat;
import java.util.ArrayList;

import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.Polyline;
import com.google.android.gms.maps.model.PolylineOptions;

import android.app.Activity;
import android.location.Location;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.Fragment;
import android.widget.TextView;

public class LogStatisticsController {
	
	GoogleMap statMap;
	LogActivity activ;
	Fragment frg;
	
	TextView tvLogstatStartTime;
	TextView tvLogstatEndTime;
	TextView tvLogstatStartAdr;
	TextView tvLogstatEndAdr;
	TextView tvLogstatFCost;
	TextView tvLogstatFCons;
	TextView tvLogstatDist;
	TextView tvLogstatTimeTaken;
	TextView tvLogstatSpeedAvg;
	
	ArrayList<TextView> allTv;
	
	private final float polylineWidth = 25.0f;
	private final int polylineColor = 0x7FFF0000;
	
	public LogStatisticsController(LogActivity activ)
	{
		this.activ = activ;
		this.frg = activ.myFragments.get(1);
		FragmentManager manager = frg.getChildFragmentManager();
		SupportMapFragment mpf = (SupportMapFragment)manager.findFragmentById(R.id.mapStats);
		statMap = mpf.getMap();
	}
	
	public void initializeDefaults()
	{
		allTv = new ArrayList<TextView>();
		
		tvLogstatStartTime = (TextView) activ.findViewById(R.id.tvLogstatStartTime);
		tvLogstatEndTime = (TextView) activ.findViewById(R.id.tvLogstatEndTime);
		tvLogstatStartAdr = (TextView) activ.findViewById(R.id.tvLogstatStartAdr);
		tvLogstatEndAdr = (TextView) activ.findViewById(R.id.tvLogstatEndAdr);
		tvLogstatFCost = (TextView) activ.findViewById(R.id.tvLogstatFCost);
		tvLogstatFCons = (TextView) activ.findViewById(R.id.tvLogstatFCons);
		tvLogstatDist = (TextView) activ.findViewById(R.id.tvLogstatDist);
		tvLogstatTimeTaken = (TextView) activ.findViewById(R.id.tvLogstatTimeTaken);
		tvLogstatSpeedAvg = (TextView) activ.findViewById(R.id.tvLogstatSpeedAvg);
		allTv.add(tvLogstatStartTime);
		allTv.add(tvLogstatEndTime);
		allTv.add(tvLogstatStartAdr);
		allTv.add(tvLogstatEndAdr);
		allTv.add(tvLogstatFCost);
		allTv.add(tvLogstatFCons);
		allTv.add(tvLogstatDist);
		allTv.add(tvLogstatTimeTaken);
		allTv.add(tvLogstatSpeedAvg);
		
		for(TextView tv : allTv)
		{
			tv.setText(R.string.str_stat_empty);
		}
	}
	
	public void loadTripData(TripModel tm)
	{
		SimpleDateFormat sdf = new SimpleDateFormat("dd.MM.yy HH:mm");
		SimpleDateFormat sdfs = new SimpleDateFormat("HH:mm");
		String startDate = sdf.format(tm.getStartTime());
		String endDate = sdf.format(tm.getEndTime());
		String timeTaken = sdfs.format(tm.getEndTime() - tm.getStartTime() - 3600000);
		
		tvLogstatStartTime.setText(startDate);
		tvLogstatEndTime.setText(endDate);
		tvLogstatStartAdr.setText(tm.getStartAddress());
		tvLogstatEndAdr.setText(tm.getEndAddress());
		tvLogstatFCost.setText(String.format("%.2f", tm.getFuelCost()) + activ.getString(R.string.str_stat_costVal));
		tvLogstatFCons.setText(String.format("%.2f", tm.getFuelConsumed()) + activ.getString(R.string.str_stat_consumedVal));
		tvLogstatDist.setText(String.format("%.2f", tm.getDistance()) + activ.getString(R.string.str_stat_distanceVal));
		tvLogstatTimeTaken.setText(timeTaken);
		tvLogstatSpeedAvg.setText(String.format("%.2f", tm.getAvgSpeed()) + activ.getString(R.string.str_stat_speedAvgVal));
		
		Trip trip = new Trip(tm);
		setupMap(trip.nodes, trip.nodesLL);
	}
	
	private void setupMap(ArrayList<Location> loclist, ArrayList<LatLng> LLlist)
	{
		if(loclist.size() > 1)
		{
			createMarkerOnCurrentPos(loclist.get(0), BitmapDescriptorFactory.HUE_GREEN, 1.0f, "START");
			
			PolylineOptions polyOpts = new PolylineOptions();
			polyOpts.width(polylineWidth);
			polyOpts.color(polylineColor);
			
			Polyline currentPolyline = statMap.addPolyline(polyOpts);
			
			createMarkerOnCurrentPos(loclist.get(loclist.size() - 1), BitmapDescriptorFactory.HUE_BLUE, 1.0f, "END");
		}
		else if(loclist.size() == 1)
		{
			createMarkerOnCurrentPos(loclist.get(0), BitmapDescriptorFactory.HUE_CYAN, 1.0f, "START AND END");
		}
		else return;
	}
	
	private boolean createMarkerOnCurrentPos(Location currentLoc, float color, float alpha, String title)
	{
		MarkerOptions startMarkerOptions = new MarkerOptions();
		startMarkerOptions.position(new LatLng(currentLoc.getLatitude(), currentLoc.getLongitude()));
		startMarkerOptions.icon(BitmapDescriptorFactory.defaultMarker(color));
		startMarkerOptions.alpha(alpha);
		startMarkerOptions.title(title);
		
		statMap.addMarker(startMarkerOptions);
		return true;
	}
}
