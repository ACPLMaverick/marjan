package com.plodz.cartracker;

import java.io.Console;

import com.google.android.gms.maps.CameraUpdate;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.model.LatLng;

import android.app.FragmentManager;
import android.location.Location;

public class MapController {
	
	GoogleMap map;
	TrackActivity activity;
	final float mapCameraZoom = 16.0f;
	
	public MapController(TrackActivity activ)
	{
		this.activity = activ;
		FragmentManager manager = activity.getFragmentManager();
		map = ((MyMapFragment)manager.findFragmentByTag("MAP")).getMap();
		centerOnUser();
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
					}
			
				});
	}
}
