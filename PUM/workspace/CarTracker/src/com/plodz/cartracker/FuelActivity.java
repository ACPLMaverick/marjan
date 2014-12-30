package com.plodz.cartracker;

import android.app.Activity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

public class FuelActivity extends Activity {
	
	FuelController controller;
	DataSource data;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_fuel);
		
		data = MainActivity.data;
		controller = new FuelController(this, data);
	}
	
	@Override
	protected void onStart()
	{
		super.onStart();
		controller.updateFieldsWithGlobals();
	}
	
	public void onUpdateFuelPricesButtonClick(View myView)
	{
		controller.updateGlobals();
	}
}
