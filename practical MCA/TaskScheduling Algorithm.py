# Function to implement Shortest Job First (SJF) Scheduling
def sjf_scheduling(jobs, service_times):
    # Combine jobs and their service times into a list of tuples
    job_service_pairs = list(zip(jobs, service_times))
    
    # Sort jobs by service time in ascending order
    job_service_pairs.sort(key=lambda x: x[1])
    
    # Initialize variables
    total_time_spent = 0
    total_waiting_time = 0
    total_turnaround_time = 0
    current_time = 0  # The time at which the current job is completed
    
    # Process each job in order
    for job, service_time in job_service_pairs:
        # Turnaround time is the current time + service time (since the job starts immediately after the previous one)
        turnaround_time = current_time + service_time
        waiting_time = current_time  # Waiting time is the time spent before starting this job
        
        # Update totals
        total_time_spent += turnaround_time
        total_waiting_time += waiting_time
        total_turnaround_time += turnaround_time
        
        # Update the current time
        current_time += service_time
        
        # Print out the details for each job
        print(f"Job {job}: Service Time = {service_time}, Waiting Time = {waiting_time}, Turnaround Time = {turnaround_time}")
    
    # Calculate average times
    avg_waiting_time = total_waiting_time / len(jobs)
    avg_turnaround_time = total_turnaround_time / len(jobs)
    avg_time_spent = total_time_spent / len(jobs)
    
    print("\nAverage Waiting Time: ", avg_waiting_time)
    print("Average Turnaround Time: ", avg_turnaround_time)
    print("Average Time Spent in System: ", avg_time_spent)

# List of jobs and their respective service times
jobs = [1, 2, 3, 4]
service_times = [5, 10, 7, 8]

# Call the SJF scheduling function
sjf_scheduling(jobs, service_times)
