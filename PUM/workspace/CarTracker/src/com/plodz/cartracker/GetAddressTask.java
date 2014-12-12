package com.plodz.cartracker;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import android.content.Context;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.os.AsyncTask;

public class GetAddressTask extends AsyncTask<Location, Void, String> {
	Context mContext;
	
	public GetAddressTask(Context context)
	{
		super();
		mContext = context;
	}
	
	@Override
	protected String doInBackground(Location... params) {
		Geocoder geo = new Geocoder(mContext, Locale.getDefault());
		Location loc = params[0];
		double lat = loc.getLatitude();
		double lon = loc.getLongitude();
		List<Address> addresses = null;
		try
		{
			addresses = geo.getFromLocation(lat,
					lon, 2);
		}
		catch(IOException e1)
		{
			e1.printStackTrace();
			return "unavailable";
		}
		catch(IllegalArgumentException e2)
		{
			e2.printStackTrace();
			return "unavailable";
		}
		
		if(addresses != null && addresses.size() > 0)
		{
			Address adr = addresses.get(0);
			String toReturn = String.format("%s, %s",
					adr.getAddressLine(0),
					adr.getAddressLine(1));
			return toReturn;
		}
		else return "unavailable";
	}

}
