package com.plodz.cartracker;

import android.app.Activity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

public class TrackActivity extends Activity {
	
	TrackingController controller;
	Trip tripToSave;
	DataSource data;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_track);
		
		data = MainActivity.data;
		controller = new TrackingController(this, data);
		controller.initialize();
	}
	
	@Override
	protected void onRestart()
	{
		super.onRestart();
		// TODO: what happens when activity has focus again?
	}
	
	@Override
	protected void onStart()
	{
		super.onStart();
		controller.connectLocalizationClient();
	}
	
	@Override
	protected void onStop()
	{
		super.onStop();
	}
	
	@Override
	protected void onDestroy()
	{
		super.onDestroy();
		// TOOD: what happens when we quit tracking?
	}
	
	public void startTracking() { controller.startTracking(); }
	
	public void endTrackingButtonClick(View v) { controller.endTracking(); }
	
	public void pauseTrackingButtonClick(View v) { controller.pauseTracking(v); }
}
