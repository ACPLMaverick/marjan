package com.plodz.cartracker;

import java.util.ArrayList;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.Map;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.KeyEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ExpandableListAdapter;
import android.widget.ExpandableListView;
import android.widget.ListView;
import android.widget.SimpleExpandableListAdapter;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.TextView.OnEditorActionListener;

import com.google.android.gms.common.*;

public class MainActivity extends Activity {
	
	public static DataSource data;
	
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        checkForGooglePlay();
        initializeGlobals();
        data = new DataSource(this);
    }
    
    @Override
    protected void onDestroy()
    {
    	super.onDestroy();
    }
    
    protected void checkForGooglePlay()
    {
    	int result = GooglePlayServicesUtil.isGooglePlayServicesAvailable(this);
    	if(result != ConnectionResult.SUCCESS) finish();
    }
    
    protected void initializeGlobals()
    {
    	// all settings will be loaded from file, saved when app closes
    	Globals.myFuelConsumption = 6.0f;				
    	Globals.myFuelType = Globals.fuelType.DIESEL;	
    	Globals.DBG_updateRatio = 0.00005f;
    	Globals.checkDelay = 1;
    	Globals.showHigherPrice = false;
    	Globals.priceDiesel = 4.8f;
    	Globals.priceDieselUltimate = 2.5f;
    	Globals.pricePB95 = 5.0f;
    	Globals.pricePB98 = 5.5f;
    	Globals.mapZoomMultiplier = 16.0f;
    	Globals.lastUpdate = new GregorianCalendar();
    }
    
    public void onStartButtonClick(View v)
    {
    	Intent intent = new Intent(this, TrackActivity.class);
    	startActivity(intent);
    }
    
    public void onLogButtonClick(View v)
    {
    	Intent intent = new Intent(this, LogActivity.class);
    	startActivity(intent);
    }
    
    public void onChartButtonClick(View v)
    {
//    	data.open();
//    	data.clearTripTable();
//    	data.close();
    }
    
    public void onPricesButtonClick(View v)
    {
    	Intent intent = new Intent(this, FuelActivity.class);
    	startActivity(intent);
    }
    
    public void onSettingsButtonClick(View v)
    {
    	Intent intent = new Intent(this, SettingsActivity.class);
    	startActivity(intent);
    }
}
