/*
 * Copyright 2024 The HAKES Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package kvlambda

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/lambda"
)

const (
	awsRegion = "us-east-1"
)

type LambdaCli struct {
	fn  string
	cli *lambda.Client
}

func (c *LambdaCli) String() string {
	return fmt.Sprintf("lambda client to function %s", c.fn)
}

func NewLambdaCli(fn string) *LambdaCli {
	return &LambdaCli{fn: fn}
}

func (c *LambdaCli) Connect() error {
	customCli := awshttp.NewBuildableClient().WithTransportOptions(func(tr *http.Transport) {
		tr.MaxIdleConnsPerHost = 128
	})
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithHTTPClient(customCli))
	if err != nil {
		return fmt.Errorf("failed to setup lambda client")
	}

	cfg.Region = awsRegion
	c.cli = lambda.NewFromConfig(cfg)
	return nil
}

func (c *LambdaCli) Invoke(input []byte) ([]byte, error) {
	log.Printf("lambda create client done: %v", time.Now().UnixMicro())

	out, err := c.cli.Invoke(context.TODO(), &lambda.InvokeInput{
		FunctionName: aws.String(c.fn),
		Payload:      input,
	})
	log.Printf("lambda invoke done: %v", time.Now().UnixMicro())
	if err != nil {
		log.Printf("failed to invoke lambda: %v", err)
		return nil, fmt.Errorf("failed to invoke")
	}
	if out.FunctionError != nil {
		log.Printf("failed to invoke lambda: %v", err)
		return nil, fmt.Errorf(string(out.Payload))
	}
	// format lambda output
	reformatPayload := out.Payload[1 : len(out.Payload)-1]
	reformatPayload = bytes.ReplaceAll(reformatPayload, []byte("\\"), []byte(""))
	return reformatPayload, nil
}
