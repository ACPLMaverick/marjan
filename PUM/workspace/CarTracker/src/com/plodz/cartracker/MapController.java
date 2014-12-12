package com.plodz.cartracker;

import java.io.Console;
import java.util.ArrayList;

import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GooglePlayServicesClient;
import com.google.android.gms.common.GooglePlayServicesUtil;
import com.google.android.gms.location.LocationClient;
import com.google.android.gms.location.LocationRequest;
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
import android.content.IntentSender;
import android.graphics.Color;
import android.location.Location;
import android.location.LocationListener;
import android.os.Bundle;

public class MapController implements
	GooglePlayServicesClient.ConnectionCallbacks,
	GooglePlayServicesClient.OnConnectionFailedListener,
	com.google.android.gms.location.LocationListener
	
{	
	private final static int CONNECTION_FAILURE_RESOLUTION_REQUEST = 9000;	
	private final static int MILISECONDS_PER_SECOND = 1000;
	private final static int UPDATE_INTERVAL_IN_SECONDS = 10;
	private final static long UPDATE_INTERVAL = MILISECONDS_PER_SECOND * UPDATE_INTERVAL_IN_SECONDS;
	private final static int FASTEST_INTERVAL_IN_SECONDS = 5;
	private final static long FASTEST_INTERVAL = MILISECONDS_PER_SECOND * FASTEST_INTERVAL_IN_SECONDS;
	
	public boolean isLoaded;
	public boolean updatesRequested;
	private boolean firstRun = true;
	public GoogleMap map;
	private TrackActivity activity;
	private LocationClient myLocClient;
	private LocationRequest myLocRequest;
	private final float mapCameraZoom = Globals.mapZoomMultiplier;
	private final float polylineWidth = 25.0f;
	private final int polylineColor = 0x7FFF0000;
	private Location startLoc;
	private Location currentLoc;
	private Location endLoc;
	private Polyline currentPolyline;
	
	public MapController(TrackActivity activ)
	{
		isLoaded = false;
		updatesRequested = false;
		this.activity = activ;
		FragmentManager manager = activity.getFragmentManager();
		map = ((MyMapFragment)manager.findFragmentByTag("MAP")).getMap();
		
		myLocClient = new LocationClient(activity, this, this);
		myLocRequest = LocationRequest.create();
		myLocRequest.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
		myLocRequest.setInterval(UPDATE_INTERVAL);
		myLocRequest.setFastestInterval(FASTEST_INTERVAL);
		
		map.setMyLocationEnabled(true);
		map.setOnMapLoadedCallback(new OnMapLoadedCallback()
		{
			@Override
			public void onMapLoaded() {
				isLoaded = true;
			}
			
		}
		);
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
	
	public void connectLocationClient()
	{
		if(myLocClient != null) myLocClient.connect();
	}
	
	public void disconnectLocationClient()
	{
		if(myLocClient != null) myLocClient.disconnect();
	}

	@Override
	public void onConnectionFailed(ConnectionResult result) {
		if(result.hasResolution())
		{
			try
			{
				result.startResolutionForResult(activity, CONNECTION_FAILURE_RESOLUTION_REQUEST);
			}
			catch (IntentSender.SendIntentException e)
			{
				e.printStackTrace();
			}
		}
		else
		{
			System.out.println("LOCCLIENT: Connection error " + String.valueOf(result.getErrorCode()));
		}
	}

	@Override
	public void onConnected(Bundle arg0) {
		System.out.println("LOCCLIENT: Connected");
		
		updatesRequested = true;
		myLocClient.requestLocationUpdates(myLocRequest, this);
	}

	@Override
	public void onDisconnected() {
		System.out.println("LOCCLIENT: Disconnected");
	}

	@Override
	public void onLocationChanged(Location location) {
		LatLng loc = new LatLng(location.getLatitude(), location.getLongitude());
		map.animateCamera(CameraUpdateFactory.newLatLngZoom(loc, mapCameraZoom));
		
		if(!isLoaded)
		{
			isLoaded = true;
			startLoc = location;
			activity.startTracking();
		}
		else currentLoc = location;
	}
}
