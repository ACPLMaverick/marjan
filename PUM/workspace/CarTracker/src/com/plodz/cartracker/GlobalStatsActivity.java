package com.plodz.cartracker;

import android.app.Activity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;

public class GlobalStatsActivity extends Activity {

	private GlobalStatsController controller;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_globalstats);
		
		controller = new GlobalStatsController(this, MainActivity.data);
		controller.initialize();
	}
	
	@Override
	protected void onResume()
	{
		super.onResume();
		
	}
}
