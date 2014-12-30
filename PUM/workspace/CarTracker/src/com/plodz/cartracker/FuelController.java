package com.plodz.cartracker;

import java.text.SimpleDateFormat;

import android.widget.TextView;

public class FuelController {
	
	private FuelActivity activity;
	private DataSource data;
	
	private TextView tvPB95;
	private TextView tvPB98;
	private TextView tvON;
	private TextView tvONU;
	private TextView tvUpdDate;
	private TextView tvUpdStatus;
	
	public FuelController(FuelActivity activity, DataSource data)
	{
		this.activity = activity;
		this.data = data;
		
		tvPB95 = (TextView) activity.findViewById(R.id.tvFuelsPB95);
		tvPB98 = (TextView) activity.findViewById(R.id.tvFuelsPB98);
		tvON = (TextView) activity.findViewById(R.id.tvFuelsON);
		tvONU = (TextView) activity.findViewById(R.id.tvFuelsONU);
		tvUpdDate = (TextView) activity.findViewById(R.id.tvFuelsLastUpdated);
		tvUpdStatus = (TextView) activity.findViewById(R.id.tvFuelsLastUpdatedText);
	}
	
	public void updateFieldsWithGlobals()
	{
		tvPB95.setText(String.format("%.2f", Globals.pricePB95) + activity.getString(R.string.str_stat_costVal));
		tvPB98.setText(String.format("%.2f", Globals.pricePB98) + activity.getString(R.string.str_stat_costVal));
		tvON.setText(String.format("%.2f", Globals.priceDiesel) + activity.getString(R.string.str_stat_costVal));
		tvONU.setText(String.format("%.2f", Globals.priceDieselUltimate) + activity.getString(R.string.str_stat_costVal));
		
		SimpleDateFormat format = new SimpleDateFormat("dd.MM.yy, HH:mm");
		
		tvUpdDate.setText(format.format(Globals.lastUpdate.getTime()));
	}
	
	public void updateGlobals()
	{
		Thread getHttpThread = new Thread(new Runnable(){

			@Override
			public void run() {
				HTTPRequestTask task = new HTTPRequestTask();
				String http = task.doInBackground(Globals.fuelURL);
				//System.out.println(http);
			}
			
		});
		getHttpThread.start();
	}
	
	private void saveGlobalsToDB()
	{
		
	}
}
