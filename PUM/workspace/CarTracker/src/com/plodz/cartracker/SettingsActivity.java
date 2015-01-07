package com.plodz.cartracker;

import java.util.ArrayList;

import com.plodz.cartracker.Globals.fuelType;

import android.app.Activity;
import android.app.Fragment;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.AdapterView.OnItemSelectedListener;

public class SettingsActivity extends Activity {

	ScrollView mainScrollView;
	Spinner spFuelType;
	EditText settingFuelConsumption;
	EditText settingUpdateRatio;
	EditText settingCheckRatio;
	ArrayAdapter<String> sfsAdapter;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_settings);
		
		// these methods should be invoked in that exact order
		initializeMainListView();
		initializeSettings();
		addItemsToFTSpinner();
	}
	
	protected void initializeMainListView()
	{
		mainScrollView = (ScrollView) findViewById(R.id.svSettings);
    	settingFuelConsumption = (EditText) findViewById(R.id.etFuelConsumption);
    	settingUpdateRatio = (EditText) findViewById(R.id.etDBGUpdateRatio);
    	settingCheckRatio = (EditText) findViewById(R.id.etDBGCheckRate);
    	spFuelType = (Spinner) findViewById(R.id.spFuelSelection);
	}
	
	protected void initializeSettings()
    {
	   	settingFuelConsumption.setText(String.valueOf(Globals.myFuelConsumption));
    	settingUpdateRatio.setText(String.valueOf(Globals.DBG_updateRatio));
    	settingCheckRatio.setText(String.valueOf(Globals.checkDelay));
    	
    	settingUpdateRatio.addTextChangedListener(new TextWatcher() {

			@Override
			public void beforeTextChanged(CharSequence s, int start, int count,
					int after) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void onTextChanged(CharSequence s, int start, int before,
					int count) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void afterTextChanged(Editable s) {
				// TODO Auto-generated method stub
				onUpdratioSettingChanged(s);
			}
    		
    	});
    	
    	settingFuelConsumption.addTextChangedListener(new TextWatcher() {

			@Override
			public void beforeTextChanged(CharSequence s, int start, int count,
					int after) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void onTextChanged(CharSequence s, int start, int before,
					int count) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void afterTextChanged(Editable s) {
				// TODO Auto-generated method stub
				onFuelconsSettingChanged(s);
			}
    		
    	});
    	
    	settingCheckRatio.addTextChangedListener(new TextWatcher() {

			@Override
			public void beforeTextChanged(CharSequence s, int start, int count,
					int after) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void onTextChanged(CharSequence s, int start, int before,
					int count) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void afterTextChanged(Editable s) {
				// TODO Auto-generated method stub
				onCheckRatioSettingChanged(s);
			}
    		
    	});
    }
    
    protected void addItemsToFTSpinner()
    {
    	String[] array = {getString(R.string.str_set_fuelTypeON), getString(R.string.str_set_fuelTypePB95), 
    			getString(R.string.str_set_fuelTypePB98), getString(R.string.str_set_fuelTypeLPG)};
    	
    	ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, R.layout.settings_listviewlayout, array);
    	
    	spFuelType.setAdapter(adapter);
    	spFuelType.setSelection(Globals.myFuelType.ordinal());
    	spFuelType.setOnItemSelectedListener(new OnItemSelectedListener() {

			@Override
			public void onItemSelected(AdapterView<?> parent, View view,
					int position, long id) {
				onSpinnerFTypeElementSelected(parent, view, position, id);
			}

			@Override
			public void onNothingSelected(AdapterView<?> parent) {
				
			}
    		
    	});
    	adapter.notifyDataSetChanged();
    }

    public void onSpinnerFTypeElementSelected(AdapterView<?> parent, View view, int position, long id)
    {
    	switch((int)id)
    	{
    	case 0:
    		Globals.myFuelType = Globals.fuelType.ON;
    		break;
    	case 1:
    		Globals.myFuelType = Globals.fuelType.PB95;
    		break;
    	case 2:
    		Globals.myFuelType = Globals.fuelType.PB98;
    		break;
    	case 3:
    		Globals.myFuelType = Globals.fuelType.LPG;
    		break;
    	default:
    		Globals.myFuelType = Globals.fuelType.ON;
    		break;
    	}
    }
    
    public void onFuelconsSettingChanged(Editable s)
    {
    	String myText = s.toString();
    	float newValue;
    	try
    	{
    		newValue = Float.valueOf(myText);
    	}
    	catch(Exception e)
    	{
    		return;
    	}
    	Globals.myFuelConsumption = newValue;
    }
    
    public void onUpdratioSettingChanged(Editable s)
    {
    	String myText = s.toString();
    	float newValue;
    	try
    	{
    		newValue = Float.valueOf(myText);
    	}
    	catch(Exception e)
    	{
    		return;
    	}
    	Globals.DBG_updateRatio = newValue;
    }
    
    public void onCheckRatioSettingChanged(Editable s)
    {
    	String myText = s.toString();
    	int newValue;
    	try
    	{
    		newValue = Integer.valueOf(myText);
    	}
    	catch(Exception e)
    	{
    		return;
    	}
    	Globals.checkDelay = newValue;
    }
}
