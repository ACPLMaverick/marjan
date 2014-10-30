package com.plodz.cartracker;

import android.app.Activity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;

public class TrackActivity extends Activity {
	
	TrackingController controller;
	
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_track);
		
		controller = new TrackingController(this);
	}
}
