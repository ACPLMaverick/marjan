package com.plodz.cartracker;

import java.text.SimpleDateFormat;
import java.util.ArrayList;

import com.google.android.gms.maps.CameraUpdate;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.CameraPosition;
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
	CameraPosition defaultPos;
	Polyline currentPolyline;
	
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
	private final float myMultiplier = 0.02f;
	private final float zoomMultiplier = Globals.mapZoomMultiplier*myMultiplier;
	
	public LogStatisticsController(LogActivity activ)
	{
		this.activ = activ;
		this.frg = activ.myFragments.get(1);
		FragmentManager manager = frg.getChildFragmentManager();
		SupportMapFragment mpf = (SupportMapFragment)manager.findFragmentById(R.id.mapStats);
		statMap = mpf.getMap();
		defaultPos = statMap.getCameraPosition();
		
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
		// reset map to default
		CameraUpdate defupd = CameraUpdateFactory.newCameraPosition(defaultPos);
		statMap.animateCamera(defupd, 1, null);
		if(currentPolyline != null) currentPolyline.remove();
		
		if(loclist.size() > 1)
		{
			createMarkerOnCurrentPos(loclist.get(0), BitmapDescriptorFactory.HUE_GREEN, 1.0f, "START");
			
			PolylineOptions polyOpts = new PolylineOptions();
			polyOpts.width(polylineWidth);
			polyOpts.color(polylineColor);
			
			currentPolyline = statMap.addPolyline(polyOpts);
			currentPolyline.setPoints(LLlist);
			
			centerCamera(LLlist);
			
			createMarkerOnCurrentPos(loclist.get(loclist.size() - 1), BitmapDescriptorFactory.HUE_BLUE, 1.0f, "END");
		}
		else if(loclist.size() == 1)
		{
			createMarkerOnCurrentPos(loclist.get(0), BitmapDescriptorFactory.HUE_CYAN, 1.0f, "START AND END");
			centerCamera(new LatLng(loclist.get(0).getLatitude(), loclist.get(0).getLongitude()));
		}
		else return;
	}
	
	private void centerCamera(ArrayList<LatLng> LLlist)
	{
		float zoom;
		LatLng center;
		
		double avgLat = 0, avgLng = 0;
		double maxLat = 0, maxLng = 0;
		double minLat = Double.MAX_VALUE, minLng = Double.MAX_VALUE;
		for(LatLng l : LLlist)
		{
			avgLat += l.latitude;
			avgLng += l.longitude;
			
			if(l.latitude > maxLat) maxLat = l.latitude;
			if(l.longitude > maxLng) maxLng = l.longitude;
			if(l.latitude < minLat) minLat = l.latitude;
			if(l.longitude < minLng) minLng = l.longitude;
		}
		avgLat /= LLlist.size();
		avgLng /= LLlist.size();
		center = new LatLng(avgLat, avgLng);
		
		double diffLat = maxLat - minLat;
		double diffLng = maxLng - minLng;
		double diff;
		if(diffLat > diffLng) diff = diffLat;
		else diff = diffLng;
		
		zoom = (float)(1/(diff != 0 ? diff : 0.0f))*zoomMultiplier;
		
		CameraUpdate pos = CameraUpdateFactory.newCameraPosition(new CameraPosition(center, zoom, 
				statMap.getCameraPosition().tilt, statMap.getCameraPosition().bearing));
		
		statMap.animateCamera(pos);
	}
	
	private void centerCamera(LatLng ll)
	{
		CameraUpdate pos = CameraUpdateFactory.newCameraPosition(new CameraPosition(ll, zoomMultiplier/myMultiplier, 
				statMap.getCameraPosition().tilt, statMap.getCameraPosition().bearing));
		
		statMap.animateCamera(pos);
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
