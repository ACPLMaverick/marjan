package com.plodz.cartracker;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.DefaultHttpClient;

import android.os.AsyncTask;

public class HTTPRequestTask extends AsyncTask<String, String, String> {

	@Override
	protected String doInBackground(String... arg0) {
		HttpClient client = new DefaultHttpClient();
		HttpGet request = new HttpGet(arg0[0]);
		HttpResponse response = null;
		try {
			response = client.execute(request);
		} catch (ClientProtocolException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return "";
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return "";
		}
		
		String toReturn = "";
		InputStream in = null;
		try {
			in = response.getEntity().getContent();
		} catch (IllegalStateException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return "";
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return "";
		}
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		StringBuilder str = new StringBuilder();
		String line = null;
		
		try
		{
			while((line = br.readLine()) != null)
			{
			    str.append(line);
			}
			in.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
			return "";
		}
		
		toReturn = str.toString();
		
		return toReturn;
	}

}
