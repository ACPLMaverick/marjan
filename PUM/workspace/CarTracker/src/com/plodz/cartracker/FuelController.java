package com.plodz.cartracker;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.GregorianCalendar;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import android.widget.TextView;

public class FuelController {
	
	private final int AWAIT_COUNTER = 100;
	
	private FuelActivity activity;
	private DataSource data;
	
	private TextView tvPB95;
	private TextView tvPB98;
	private TextView tvON;
	private TextView tvLPG;
	private TextView tvUpdDate;
	private TextView tvUpdStatus;
	private TextView tvProvider;
	
	private String http;
	
	public FuelController(FuelActivity activity, DataSource data)
	{
		this.activity = activity;
		this.data = data;
		
		tvPB95 = (TextView) activity.findViewById(R.id.tvFuelsPB95);
		tvPB98 = (TextView) activity.findViewById(R.id.tvFuelsPB98);
		tvON = (TextView) activity.findViewById(R.id.tvFuelsON);
		tvLPG = (TextView) activity.findViewById(R.id.tvFuelsLPG);
		tvUpdDate = (TextView) activity.findViewById(R.id.tvFuelsLastUpdated);
		tvUpdStatus = (TextView) activity.findViewById(R.id.tvFuelsLastUpdatedText);
		tvProvider = (TextView) activity.findViewById(R.id.tvFuelsProvider);
		
		tvProvider.setText(Globals.fuelURL);
	}
	
	public void updateFieldsWithGlobals()
	{
		tvPB95.setText(String.format("%.2f", Globals.pricePB95) + activity.getString(R.string.str_stat_costVal));
		tvPB98.setText(String.format("%.2f", Globals.pricePB98) + activity.getString(R.string.str_stat_costVal));
		tvON.setText(String.format("%.2f", Globals.priceON) + activity.getString(R.string.str_stat_costVal));
		tvLPG.setText(String.format("%.2f", Globals.priceLPG) + activity.getString(R.string.str_stat_costVal));
		
		SimpleDateFormat format = new SimpleDateFormat("dd.MM.yy, HH:mm");
		tvUpdDate.setText(format.format(Globals.lastUpdate.getTime()));
	}
	
	public void updateGlobals()
	{
		Runnable upd = new Runnable()
		{
			public void run()
			{
				tvUpdDate.setText("");
				tvUpdStatus.setText(activity.getString(R.string.str_fuels_updating));
			}
		};
		activity.runOnUiThread(upd);
		
		Thread getHttpThread = new Thread(new Runnable(){

			@Override
			public void run() {
				HTTPRequestTask task = new HTTPRequestTask();
				http = task.doInBackground(Globals.fuelURL);
			}
			
		});
		getHttpThread.start();
		
		for(int i = 0; i < AWAIT_COUNTER; i++)
		{
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				return;
			}
			if(http != null) break;
		}
		
		if(http == null)
		{
			tvUpdStatus.setText(activity.getString(R.string.str_fuels_error));
			return;
		}
		
		Document doc = Jsoup.parse(http);
		boolean parsed = parseEPetrol(doc);
		if(parsed)
		{
			tvUpdStatus.setText(activity.getString(R.string.str_fuels_lastUpdated));
			GregorianCalendar cal = new GregorianCalendar();
			cal.setTime(new Date());
			Globals.lastUpdate = cal;
			updateFieldsWithGlobals();
			savePricesToDB();
		}
		else
		{
			tvUpdStatus.setText(activity.getString(R.string.str_fuels_error));
		}
	}
	
	private boolean parseEPetrol(Document doc)
	{
		float prLPG, prON, prPB95, prPB98;
		try
		{
			Elements evens = doc.getElementsByClass("even");
			Element globalPrice = evens.first();
			Elements priceElems = globalPrice.getAllElements();
			
			String strPB98, strPB95, strON, strLPG;
			strPB98 = priceElems.get(3).text();
			strPB95 = priceElems.get(4).text();
			strON = priceElems.get(5).text();
			strLPG = priceElems.get(6).text();
			
			strPB98 = strPB98.replaceAll(",",".");
			strPB95 = strPB95.replaceAll(",",".");
			strON = strON.replaceAll(",",".");
			strLPG = strLPG.replaceAll(",",".");
			
			prLPG = Float.valueOf(strLPG);
			prON = Float.valueOf(strON);
			prPB95 = Float.valueOf(strPB95);
			prPB98 = Float.valueOf(strPB98);
		}
		catch(Exception e)
		{
			e.printStackTrace();
			return false;
		}
		
		Globals.priceLPG = prLPG;
		Globals.priceON = prON;
		Globals.pricePB95 = prPB95;
		Globals.pricePB98 = prPB98;
		
		return true;
	}
	
	private void savePricesToDB()
	{
		data.open();
		data.saveFuels();
		data.close();
	}
}
