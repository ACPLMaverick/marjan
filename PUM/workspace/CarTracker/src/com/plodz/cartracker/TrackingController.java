package com.plodz.cartracker;

import java.util.ArrayList;

import android.location.Location;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.Polyline;

public class TrackingController {
	
	public static final int STAT_COUNT = 7;
	
	public boolean ifUpdate = true;
	
	private MapController mc;
	private TrackActivity activity;
	private DataSource data;
	private Trip trip;
	private double lastLat;
	private double lastLon;
	private Thread checkThread;
	private boolean ifRunning = true;
	
	private TextView statConsTxt;
	private TextView statCostTxt;
	private TextView statDistTxt;
	private TextView statSpdATxt;
	private TextView statSpdCTxt;
	private TextView statTimeSTxt;
	private TextView statTimeCTxt;
	
	private TextView dbgTxt1;
	
	public TrackingController(TrackActivity activity, DataSource data)
	{
		this.activity = activity;
		this.data = data;
	}
	
	public void initialize()
	{
		mc = new MapController(activity);
		trip = new Trip();
		
		statConsTxt = (TextView) activity.findViewById(R.id.statConsumedFuelTxt);
		statCostTxt = (TextView) activity.findViewById(R.id.statCostTxt);
		statDistTxt = (TextView) activity.findViewById(R.id.statDistanceTxt);
		statSpdATxt = (TextView) activity.findViewById(R.id.statSpeedAvgTxt);
		statSpdCTxt = (TextView) activity.findViewById(R.id.statSpeedCurrentTxt);
		statTimeSTxt = (TextView) activity.findViewById(R.id.statTimeStartedTxt);
		statTimeCTxt = (TextView) activity.findViewById(R.id.statTimeCurrentTxt);
		
		String empty = activity.getResources().getText(R.string.str_stat_empty).toString();
		updateStatsInView(new String[] {empty, empty, empty, empty, empty, empty, empty });
		
		dbgTxt1 = (TextView) activity.findViewById(R.id.dbgTxt1);
	}
	
	public void connectLocalizationClient()
	{
		if(mc != null) mc.connectLocationClient();
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
				for(;ifRunning == true;)
				{
					try {
						Thread.sleep(Globals.checkDelay*1000);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
						break;
					}
					if(ifUpdate)
					{
						if(Math.abs(lastLat - mc.getMyLocation().getLatitude()) > Globals.DBG_updateRatio ||
								Math.abs(lastLon - mc.getMyLocation().getLongitude()) > Globals.DBG_updateRatio)
						{
							lastLat = mc.getMyLocation().getLatitude();
							lastLon = mc.getMyLocation().getLongitude(); 
							trip.update(mc.getMyLocation(), Globals.checkDelay*1000);
							
							activity.runOnUiThread(new Runnable(){

								@Override
								public void run() {
									mc.updatePolyline(trip.nodesLL);
								}
								
							});
						}
						else
						{
							trip.updateTimeOnly(Globals.checkDelay*1000);
						}
						
						activity.runOnUiThread(new Runnable()
						{
							@Override
							public void run()
							{
								dbgTxt1.setText(String.valueOf(trip.nodes.size()));
								updateStatsInView(trip.getStatsAsStringArray());
							}
						});
					}			
				}
			}
			
		});
		checkThread.start();
	}
	
	public void endTracking()
	{
		saveTripToDB();
		
		ifRunning = false;
		
		mc.disconnectLocationClient();
		
		activity.finish();
	}
	
	public void pauseTracking(View v)
	{
			if(ifUpdate)
			{
				Thread blocker = new Thread(new Runnable(){

					@Override
					public void run() {
						// TODO Auto-generated method stub
						synchronized(checkThread)
						{
							try {
								checkThread.wait();
							} catch (InterruptedException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
								return;
							}
						}
					}
					
				});
				ifUpdate = false;
				Button b = (Button) v;
				b.setText(R.string.str_button_resumeTrack);
			}
			else
			{
				Thread relaser = new Thread(new Runnable(){

					@Override
					public void run() {
						// TODO Auto-generated method stub
						synchronized(checkThread)
						{
							checkThread.notifyAll();
						}
					}
					
				});
				ifUpdate = true;
				Button b = (Button) v;
				b.setText(R.string.str_button_pauseTrack);
			}
	}
	
	private void saveTripToDB()
	{
		TripModel tm = new TripModel(trip);
		data.open();
		data.addTripModel(tm);
		data.close();
	}
	
	private void updateStatsInView(String[] values)
	{	
		statConsTxt.setText(activity.getResources().getText(R.string.str_stat_consumed) + values[0] 
				+ activity.getResources().getText(R.string.str_stat_consumedVal));
		statCostTxt.setText(activity.getResources().getText(R.string.str_stat_cost) + values[1] 
				+ activity.getResources().getText(R.string.str_stat_costVal));
		statDistTxt.setText(activity.getResources().getText(R.string.str_stat_distance) + values[2] 
				+ activity.getResources().getText(R.string.str_stat_distanceVal));
		statSpdATxt.setText(activity.getResources().getText(R.string.str_stat_speedAvg) + values[3] 
				+ activity.getResources().getText(R.string.str_stat_speedAvgVal));
		statSpdCTxt.setText(activity.getResources().getText(R.string.str_stat_speedCurr) + values[4] 
				+ activity.getResources().getText(R.string.str_stat_speedCurrVal));
		statTimeSTxt.setText(activity.getResources().getText(R.string.str_stat_timeStart) + values[5] 
				+ activity.getResources().getText(R.string.str_stat_timeStartVal));
		statTimeCTxt.setText(activity.getResources().getText(R.string.str_stat_timeCurr) + values[6] 
				+ activity.getResources().getText(R.string.str_stat_timeCurrVal));
	}
}
