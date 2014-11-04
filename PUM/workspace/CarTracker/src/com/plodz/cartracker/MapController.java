package com.plodz.cartracker;

import java.io.Console;
import java.util.ArrayList;

import com.google.android.gms.maps.CameraUpdate;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.GoogleMap.OnMapLoadedCallback;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.Polyline;
import com.google.android.gms.maps.model.PolylineOptions;

import android.app.FragmentManager;
import android.graphics.Color;
import android.location.Location;

public class MapController {
	
	public boolean isLoaded;
	private boolean firstRun = true;
	public GoogleMap map;
	private TrackActivity activity;
	private final float mapCameraZoom = 16.0f;
	private final float polylineWidth = 25.0f;
	private final int polylineColor = 0x7FFF0000;
	private Location startLoc;
	private Location currentLoc;
	private Location endLoc;
	private Polyline currentPolyline;
	
	public MapController(TrackActivity activ)
	{
		isLoaded = false;
		this.activity = activ;
		FragmentManager manager = activity.getFragmentManager();
		map = ((MyMapFragment)manager.findFragmentByTag("MAP")).getMap();
		
		map.setOnMapLoadedCallback(new OnMapLoadedCallback()
		{
			@Override
			public void onMapLoaded() {
				// TODO Auto-generated method stub
				isLoaded = true;
			}
			
		}
		);
	}
	
	public void centerOnUser()
	{
		map.setMyLocationEnabled(true);
		map.setOnMyLocationChangeListener(new GoogleMap.OnMyLocationChangeListener()
				{

					@Override
					public void onMyLocationChange(Location arg0) {
						LatLng location = new LatLng(arg0.getLatitude(), arg0.getLongitude());
						map.animateCamera(CameraUpdateFactory.newLatLngZoom(location, mapCameraZoom));
						if(!isLoaded)
						{
							isLoaded = true;
							startLoc = arg0;
							activity.startTracking();
						}
						else currentLoc = arg0;
					}
			
				});
	}
	
	public boolean createMarkerOnCurrentPos(float color, float alpha)
	{
		if(firstRun) 
		{
			currentLoc = startLoc;
			firstRun = false;
		}
		MarkerOptions startMarkerOptions = new MarkerOptions();
		startMarkerOptions.position(new LatLng(currentLoc.getLatitude(), currentLoc.getLongitude()));
		startMarkerOptions.icon(BitmapDescriptorFactory.defaultMarker(color));
		startMarkerOptions.alpha(alpha);
		startMarkerOptions.title("START");
		
		map.addMarker(startMarkerOptions);
		return true;
	}
	
	public void initializePolyline()
	{
		PolylineOptions polyOpts = new PolylineOptions();
//		polyOpts.add(new LatLng(currentLoc.getLatitude(), currentLoc.getLongitude()));
//		polyOpts.add(new LatLng(currentLoc.getLatitude() + 1, currentLoc.getLongitude() + 1));
		polyOpts.width(polylineWidth);
		polyOpts.color(polylineColor);
		
		currentPolyline = map.addPolyline(polyOpts);
	}
	
	public void updatePolyline(ArrayList<LatLng> points) throws NullPointerException
	{
		if(currentPolyline == null) throw new NullPointerException();
		currentPolyline.setPoints(points);
	}
	
	public Location getMyLocation() { return currentLoc; }
}
