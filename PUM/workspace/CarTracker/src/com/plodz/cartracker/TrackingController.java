package com.plodz.cartracker;

import java.util.ArrayList;

import android.location.Location;
import android.os.Handler;
import android.os.Looper;
import android.widget.TextView;

import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.Polyline;

public class TrackingController {
	
	private MapController mc;
	private TrackActivity activity;
	ArrayList<LatLng> routePoints;
	double lastLat;
	double lastLon;
	double distCheckRes = 0.00005;
	Thread checkThread;
	
	TextView dbgTxt1;
	
	public TrackingController(TrackActivity activity)
	{
		this.activity = activity;
	}
	
	public void initialize()
	{
		mc = new MapController(activity);
		mc.centerOnUser();
		routePoints = new ArrayList<LatLng>();
		dbgTxt1 = (TextView) activity.findViewById(R.id.dbgTxt1);
	}
	
	public void startTracking()
	{
		mc.createMarkerOnCurrentPos(BitmapDescriptorFactory.HUE_GREEN, 1.0f);
		mc.initializePolyline();
		lastLat = mc.getMyLocation().getLatitude();
		lastLon = mc.getMyLocation().getLongitude();
		checkThread = new Thread(new Runnable()
		{

			@Override
			public void run() {
				for(;;)
				{
					try {
						Thread.sleep(1000);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
						break;
					}
					
					activity.runOnUiThread(new Runnable()
					{
						@Override
						public void run()
						{
							dbgTxt1.setText("NODE COUNT: " + String.valueOf(routePoints.size()));
						}
					});
					
					if(Math.abs(lastLat - mc.getMyLocation().getLatitude()) > distCheckRes ||
							Math.abs(lastLon - mc.getMyLocation().getLongitude()) > distCheckRes)
					{
						lastLat = mc.getMyLocation().getLatitude();
						lastLon = mc.getMyLocation().getLongitude(); 

						routePoints.add(new LatLng(mc.getMyLocation().getLatitude(), mc.getMyLocation().getLongitude()));
						
						activity.runOnUiThread(new Runnable(){

							@Override
							public void run() {
								mc.updatePolyline(routePoints);
							}
							
						});
						
						//mc.updatePolyline(routePoints);
						System.out.println(String.valueOf(routePoints.size()));
					}
				}
			}
			
		});
		checkThread.start();
	}
	
	public void endTracking()
	{
		
	}
}
